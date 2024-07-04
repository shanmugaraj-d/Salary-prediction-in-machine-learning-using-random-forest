Here's the README file for your project:

---

# Salary Prediction in Machine Learning using Random Forest

## AIM:

In this project, we predict whether a person’s income is above 50k or below 50k using various features like age, education, and occupation. The dataset used is the Adult Census Income dataset from Kaggle, containing about 32,561 rows and 15 features.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Steps](#steps)
  - [Step 0: Import Libraries and Dataset](#step-0-import-libraries-and-dataset)
  - [Step 1: Descriptive Analysis](#step-1-descriptive-analysis)
  - [Step 2: Exploratory Data Analysis](#step-2-exploratory-data-analysis)
  - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
  - [Step 4: Data Modelling](#step-4-data-modelling)
  - [Step 5: Model Evaluation](#step-5-model-evaluation)
  - [Step 6: Hyperparameter Tuning](#step-6-hyperparameter-tuning)
- [Summary](#summary)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Salary-prediction-in-machine-learning-using-random-forest.git
    cd Salary-prediction-in-machine-learning-using-random-forest
    ```
2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the script:
    ```sh
    python salary_prediction.py
    ```

## Steps

### Step 0: Import Libraries and Dataset

All the standard libraries like numpy, pandas, matplotlib, and seaborn are imported in this step. We use numpy for linear algebra operations, pandas for data frames, matplotlib, and seaborn for plotting graphs. The dataset is imported using the pandas command `read_csv()`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('adult.csv')
```

### Step 1: Descriptive Analysis

```python
# Preview dataset
dataset.head()
# Shape of dataset
print('Rows: {} Columns: {}'.format(dataset.shape[0], dataset.shape[1]))
# Features data-type
dataset.info()
# Statistical summary
dataset.describe().T
# Check for null values
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
# Check for '?' in dataset
round((dataset.isin(['?']).sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
# Checking the counts of label categories
income = dataset['income'].value_counts(normalize=True)
round(income * 100, 2).astype('str') + ' %'
```

Observations:
1. The dataset doesn’t have any null values but contains missing values in the form of ‘?’ which needs to be preprocessed.
2. The dataset is unbalanced, with 75.92% values having income less than 50k, and 24.08% values having income more than 50k.

### Step 2: Exploratory Data Analysis

#### 2.1 Univariate Analysis
![Univariate Analysis](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/059e26c9-c428-42c5-883a-53e5d7d73be5)

#### 2.2 Bivariate Analysis
![Bivariate Analysis](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/3deafd40-09af-4a00-a89e-58cba5272fd8)

#### 2.3 Multivariate Analysis
![Multivariate Analysis](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/e147caed-4dc1-4486-9007-d503df1ddfb7)

Observations:
1. Most people in the dataset are young, white, male, high school graduates with 9 to 10 years of education, and work 40 hours per week.
2. The dependent feature ‘income’ is highly correlated with age, years of education, capital gain, and hours per week.

### Step 3: Data Preprocessing

```python
dataset = dataset.replace('?', np.nan)
# Checking null values
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
columns_with_nan = ['workclass', 'occupation', 'native.country']
for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)
# Encoding object columns
from sklearn.preprocessing import LabelEncoder
for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])
# Splitting dataset into X and Y
X = dataset.drop('income', axis=1)
Y = dataset['income']
# Feature selection using ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(random_state=42)
selector.fit(X, Y)
feature_imp = selector.feature_importances_
for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))
X = X.drop(['workclass', 'education', 'race', 'sex', 'capital.loss', 'native.country'], axis=1)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
for col in X.columns:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
# Oversampling using RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
ros.fit(X, Y)
X_resampled, Y_resampled = ros.fit_resample(X, Y)
# Splitting into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
```

### Step 4: Data Modelling

```python
from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state=42)
ran_for.fit(X_train, Y_train)
Y_pred_ran_for = ran_for.predict(X_test)
```

Random Forest is a Supervised learning algorithm used for both classification and regression. It is a type of bagging ensemble algorithm, creating multiple decision trees simultaneously. The final prediction is selected using majority voting.

![Random Forest](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/219c2ed3-ac44-47bc-a408-77dc4eb36215)

### Step 5: Model Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score
print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))
```

### Step 6: Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
max_depth = [int(x) for x in np.linspace(40, 150, num=15)]
param_dist = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
}
rf_tuned = RandomForestClassifier(random_state=42)
rf_cv = RandomizedSearchCV(estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)
rf_cv.fit(X_train, Y_train)
print(rf_cv.best_score_)
print(rf_cv.best_params_)
rf_best = RandomForestClassifier(max_depth=102, n_estimators=40, random_state=42)
rf_best.fit(X_train, Y_train)
Y_pred_rf_best = rf_best.predict(X_test)
print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_rf_best) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_rf_best) * 100, 2))
```

```python
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_pred_rf_best)
print(classification_report(Y_test, Y_pred_rf_best))
```

## Summary

The Random Forest Classifier model achieves 93% accuracy on the testing data. This is a simple and beginner-friendly Random Forest Classifier model.

## Conclusion

This project proposes a salary prediction system using a random forest algorithm. The system's result is calculated by comparing it with other algorithms in terms of standard scores and curves like classification accuracy, F1 score, and the ROC curve
