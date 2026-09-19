import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
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

df = pd.read_csv("titanic.csv")
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

clv_results = []
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
adj_r2 = 1 - (1 - r2) * (len(yreg_test) - 1) / (len(yreg_test) - Xreg_trains.shape[1] - 1)

joblib.dump(best_pipe, "titanic_pipeline.joblib")


loaded_pipe = joblib.load("titanic_pipeline.joblib")
sample_raw = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 28.0,
            "sibsp": 0,
            "parch": 0,
            "embarked": "C",
            "class": "First",
            "who": "woman",
        }
    ]
)

print(
    f"\nSample Prediction (0=Perished, 1=Survived): {loaded_pipe.predict(sample_raw)[0]}"
)