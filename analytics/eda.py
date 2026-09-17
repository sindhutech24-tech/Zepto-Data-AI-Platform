import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


csv_path = "titanic.csv"
if not os.path.exists(csv_path):
    df = sns.load_dataset("titanic")
    df.to_csv(csv_path, index=False)
    print(f"Dataset cached locally to {csv_path}")
else:
    print(f"Loading dataset from local: {csv_path}")
    df = pd.read_csv(csv_path)

df.info()

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


print(df.describe())

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100
missing_df = pd.DataFrame(
    {"Missing Count": missing_count, "Percentage (%)": missing_percent}
)
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
    by="Percentage (%)", ascending=False


df["deck"] = df["deck"].astype(str).replace("nan", "Unknown")


df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
    lambda x: x.fillna(x.median())
)


df.dropna(subset=["embarked", "embark_town"], inplace=True)

if "embark_town" in df.columns:
    df.drop(columns=["embark_town"], inplace=True)

print(
    f"Cleaned DataFrame Shape: {df.shape} | Remaining Missing Values: {df.isnull().sum().sum()}"
)


def identify_outliers_iqr(col_data, col_name):
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = col_data[
        (col_data < lower_bound) | (col_data > upper_bound)
    ]
    print(f"[{col_name}] Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(
        f"[{col_name}] Outlier Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
    )
    print(f"[{col_name}] Total Outliers: {len(outliers)}")
    return outliers


print("\n--- Age Analysis ---")
identify_outliers_iqr(df["age"], "Age")

print("\n--- Fare Analysis ---")
identify_outliers_iqr(df["fare"], "Fare")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print(f"Fare Mean:   {fare_mean:.2f}")
print(f"Fare Median: {fare_median:.2f}")
print(f"Fare Mode:   {fare_mode:.2f}")


fig, axes = plt.subplots(2, 2, figsize=(12, 8))


sns.histplot(df["age"], kde=True, ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Age Distribution (Histogram + KDE)")
sns.boxplot(x=df["age"], ax=axes[0, 1], color="lightblue")
axes[0, 1].set_title("Age Box Plot")


sns.histplot(df["fare"], kde=True, ax=axes[1, 0], color="salmon")
axes[1, 0].set_title("Fare Distribution (Histogram + KDE)")
sns.boxplot(x=df["fare"], ax=axes[1, 1], color="lightcoral")
axes[1, 1].set_title("Fare Box Plot")

plt.tight_layout()
plt.savefig("univariate_analysis.png")
plt.close()


male_mask = df["sex"] == "male"
female_mask = df["sex"] == "female"
print(f"Female Survival Rate: {df[female_mask]['survived'].mean():.2%}")
print(f"Male Survival Rate:   {df[male_mask]['survived'].mean():.2%}")

for pclass_val in sorted(df["pclass"].unique()):
    pclass_mask = df["pclass"] == pclass_val
    sr = df[pclass_mask]["survived"].mean()
    print(f"Pclass {pclass_val} Survival Rate: {sr:.2%}")


print("\n--- Survival Rate by Sex AND Pclass ---")
for p_val in sorted(df["pclass"].unique()):
    for s_val in ["female", "male"]:
        combined_mask = (df["pclass"] == p_val) & (df["sex"] == s_val)
        sr = df[combined_mask]["survived"].mean()
        print(f"Pclass {p_val} | {s_val.capitalize()}: {sr:.2%}")


num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1
)
plt.title("6x6 Numeric Feature Correlation Matrix")
plt.savefig("correlation_heatmap.png")
plt.close()

corr_pairs = (
    corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
)
top_2_corr = corr_pairs[corr_pairs < 1.0].head(2)

for pair, val in top_2_corr.items():
    actual_val = corr_matrix.loc[pair[0], pair[1]]
    print(f"Pair {pair}: r = {actual_val:.3f} (|r| = {val:.3f})")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

sns.barplot(
    data=df, x="pclass", y="survived", hue="sex", ci=None, ax=axes[0, 0]
)
axes[0, 0].set_title("1. Survival Rate Passenger Class and Gender")
axes[0, 0].set_ylabel("Survival Rate")


sns.boxplot(
    data=df,
    x="pclass",
    y="age",
    hue="survived",
    palette="Set2",
    ax=axes[0, 1],
)
axes[0, 1].set_title("2. Age Distribution by Class and Survival")


sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7,
    palette="coolwarm",
    ax=axes[1, 0],
)
axes[1, 0].set_title("3. Fare vs. Age by Survival Status")
axes[1, 0].set_yscale("log")  # Log scale due to extreme fare skewness
axes[1, 0].set_ylabel("Fare (Log Scale)")


df["family_size"] = df["sibsp"] + df["parch"] + 1
sns.lineplot(
    data=df, x="family_size", y="survived", marker="o", ax=axes[1, 1]
)
axes[1, 1].set_title("4. Survival Rate by Total Family Size")
axes[1, 1].set_xlabel("Family Size (SibSp + Parch + 1)")
axes[1, 1].set_ylabel("Survival Rate")

plt.tight_layout()
plt.savefig("multivariate_data_story.png")
plt.close()



scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[["age", "fare"]])
df_scaled = pd.DataFrame(
    scaled_features, columns=["age_z", "fare_z"], index=df.index
)

print("Before Scaling:")
print(
    f"Age  -> Mean: {df['age'].mean():.4f}, Std: {df['age'].std():.4f}"
)
print(
    f"Fare -> Mean: {df['fare'].mean():.4f}, Std: {df['fare'].std():.4f}"
)

print("\nAfterScaling:")
print(
    f"Age_z  -> Mean: {df_scaled['age_z'].mean():.4f}, Std: {df_scaled['age_z'].std():.4f}"
)
print(
    f"Fare_z -> Mean: {df_scaled['fare_z'].mean():.4f}, Std: {df_scaled['fare_z'].std():.4f}"
)

