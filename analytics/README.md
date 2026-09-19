# Analytics Module

This folder contains the data exploration and machine learning workflow for the Titanic survival analysis used in the Zepto Data AI Platform project.

## Objective

The analytics module is designed to:

- load and inspect the Titanic dataset
- clean and transform the data for analysis
- perform exploratory data analysis (EDA)
- investigate feature relationships and survival patterns
- train and compare classification models
- train a regression model for fare prediction
- save model artifacts and visual outputs for reporting


## Dataset

The dataset used in this module is the classic Titanic passenger dataset.

- Source: Seaborn's `titanic` dataset when the local copy is not available
- Local cache: `analytics/data/titanic.csv`
- Key columns include: `survived`, `pclass`, `sex`, `age`, `fare`, `embarked`, `sibsp`, `parch`, and `class`

If the CSV file is not already present, `eda.py` automatically downloads it from the Seaborn dataset and saves it locally.

## Files

### `eda.py`

Performs the full exploratory analysis workflow:

- loads the dataset
- checks data shape and summary statistics
- identifies missing values and percentages
- fills missing age values by passenger class and sex
- drops incomplete rows with missing keys
- analyzes outliers for age and fare
- creates univariate and multivariate plots
- studies survival rates across sex and class
- computes correlation metrics
- saves visual reports to the analytics folder

### `modeling.py`

Builds predictive models for the Titanic dataset:

- defines preprocessing pipelines for numeric and categorical features
- trains multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- tunes the Random Forest with `GridSearchCV`
- evaluates accuracy, precision, recall, F1-score, and AUC
- plots ROC and confusion matrix results
- trains a linear regression model to predict `fare`
- stores the best pipeline as `models/titanic_pipeline.joblib`
- saves visual evaluation outputs in the `models/` folder

### `requirements.txt`

Python dependencies required for the analytics workflow:

```text
pandas
numpy
seaborn
matplotlib
scikit-learn
joblib
imblearn
```

## Setup

From the repository root:

```bash
cd analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Analysis

### Exploratory data analysis

```bash
python eda.py
```

This script generates charts such as:

- `univariate_analysis.png`
- `correlation_heatmap.png`
- `multivariate_data_story.png`

### Model training and evaluation

```bash
python modeling.py
```

This script generates and saves:

- `models/roc_curves.png`
- `models/confusion_matrix.png`
- `models/regression_actual_vs_pred.png`
- `models/titanic_pipeline.joblib`