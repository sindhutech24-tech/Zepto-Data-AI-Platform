from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "data" / "titanic.csv")
X = df.drop(
    columns=[
        "survived",
        "alive",
        "embark_town",
        "adult_male",
        "alone",
        "deck",
    ],
    errors="ignore",
)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["sex", "pclass", "embarked", "class", "who"]

preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

clf_results = []
roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    clf_results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "AUC": roc_auc_score(y_test, probs),
        }
    )
    RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=roc_ax)

roc_ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
roc_ax.set_title("ROC curves: baseline classifiers (test set)")
roc_fig.tight_layout()
roc_fig.savefig(MODEL_DIR / "roc_curves.png", dpi=150)
plt.close(roc_fig)

results_df = pd.DataFrame(clf_results).set_index("Model").round(4)
print(results_df)
results_df.plot(kind="bar", figsize=(8, 5), rot=0, ylim=(0, 1))
plt.title("Classifier comparison (test set)")
plt.tight_layout()
plt.close()


param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [3, 5, None],
}
rf_pipe = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(oob_score=True, random_state=42)),
    ]
)
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="f1").fit(X_train, y_train)

best_pipe = grid.best_estimator_
print(f"Best Params: {grid.best_params_}")
print(f"OOB Score: {best_pipe.named_steps['clf'].oob_score_:.4f}")

best_preds = best_pipe.predict(X_test)
print(
    f"Tuned RF test: Acc={accuracy_score(y_test, best_preds):.4f} "
    f"Prec={precision_score(y_test, best_preds):.4f} "
    f"Rec={recall_score(y_test, best_preds):.4f} "
    f"F1={f1_score(y_test, best_preds):.4f} "
    f"AUC={roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1]):.4f}"
)

cm_fig, cm_ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_preds, display_labels=["Perished", "Survived"], ax=cm_ax, cmap="Blues"
)
cm_ax.set_title("Tuned Random Forest: confusion matrix (test)")
cm_fig.tight_layout()
cm_fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
plt.close(cm_fig)

importances = pd.Series(
    best_pipe.named_steps["clf"].feature_importances_,
    index=best_pipe.named_steps["prep"].get_feature_names_out(),
).sort_values()
print(importances.tail(8)[::-1].round(4))
importances.tail(12).plot(kind="barh", figsize=(7, 6))
plt.title("Tuned Random Forest: top 12 feature importances")
plt.tight_layout()
plt.close()


X_reg = df.drop(
    columns=["fare", "alive", "embark_town", "adult_male", "alone", "deck"],
    errors="ignore",
)
y_reg= df["fare"]
Xreg_train, Xreg_test, yreg_train, yreg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_cols_num = ["age", "sibsp", "parch", "survived"]
reg_prep = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
            ),
            reg_cols_num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

reg_pipe = Pipeline([("prep", reg_prep), ("reg", LinearRegression())])
reg_pipe.fit(Xreg_train, yreg_train)
r_preds = reg_pipe.predict(Xreg_test)

r2 = r2_score(yreg_test, r_preds)
adj_r2 = 1 - (1 - r2) * (len(yreg_test) - 1) / (len(yreg_test) - Xreg_train.shape[1] - 1)
print(
    f"R2={r2:.4f} AdjR2={adj_r2:.4f} "
    f"MAE={mean_absolute_error(yreg_test, r_preds):.2f} "
    f"RMSE={mean_squared_error(yreg_test, r_preds) ** 0.5:.2f}"
)

plt.figure(figsize=(5.5, 5))
plt.scatter(yreg_test, r_preds, alpha=0.5)
lim = [min(yreg_test.min(), r_preds.min()), max(yreg_test.max(), r_preds.max())]
plt.plot(lim, lim, "r--", label="Perfect prediction")
plt.xlabel("Actual fare")
plt.ylabel("Predicted fare")
plt.title("Linear regression: actual vs predicted fare (test)")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "regression_actual_vs_pred.png", dpi=150)
plt.close()

joblib.dump(best_pipe, MODEL_DIR / "titanic_pipeline.joblib")


loaded_pipe = joblib.load(MODEL_DIR / "titanic_pipeline.joblib")
sample_raw = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 28.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C",
            "class": "First",
            "who": "woman",
        }
    ]
)

print(
    f"\nSample Prediction (0=Perished, 1=Survived): {loaded_pipe.predict(sample_raw)[0]}"
)
