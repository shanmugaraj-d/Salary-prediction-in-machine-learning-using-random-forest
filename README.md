# Salary-prediction-in-machine-learning-using-random-forest

AIM:
   In this project, we are going to predict whether a person’s income is above 50k or below 50k using various features like age, education, and occupation. The dataset we are going to use is the Adult census income dataset from Kaggle which contains about 32561 rows and 15 features.

Step 0: Import libraries and dataset

All the standard libraries like numpy, pandas, matplotlib, and seaborn are imported in this step. We use numpy for linear algebra operations, pandas for using data frames, matplotlib, and seaborn for plotting graphs. The dataset is imported using the pandas command read_csv().

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importing dataset
dataset = pd.read_csv('adult.csv')

Step 1: Descriptive analysis

# Preview dataset
dataset.head()

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/872cd897-f585-49e7-8d1c-e91f8ebfa8e6)

# Shape of dataset
print('Rows: {} Columns: {}'.format(dataset.shape[0], dataset.shape[1]))
Output:
Rows: 32561 Columns: 15
# Features data-type 
dataset.info()
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             32561 non-null  int64 
 1   workclass       32561 non-null  object
 2   fnlwgt          32561 non-null  int64 
 3   education       32561 non-null  object
 4   education.num   32561 non-null  int64 
 5   marital.status  32561 non-null  object
 6   occupation      32561 non-null  object
 7   relationship    32561 non-null  object
 8   race            32561 non-null  object
 9   sex             32561 non-null  object
 10  capital.gain    32561 non-null  int64 
 11  capital.loss    32561 non-null  int64 
 12  hours.per.week  32561 non-null  int64 
 13  native.country  32561 non-null  object
 14  income          32561 non-null  object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
# Statistical summary
dataset.describe().T

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/8772ef2f-fe43-4825-85b6-7fa94730d428)

# Check for null values
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
Output:
age               0.0 %
workclass         0.0 %
fnlwgt            0.0 %
education         0.0 %
education.num     0.0 %
marital.status    0.0 %
occupation        0.0 %
relationship      0.0 %
race              0.0 %
sex               0.0 %
capital.gain      0.0 %
capital.loss      0.0 %
hours.per.week    0.0 %
native.country    0.0 %
income            0.0 %
dtype: object
# Check for '?' in dataset
round((dataset.isin(['?']).sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
Output:
age                0.0 %
workclass         5.64 %
fnlwgt             0.0 %
education          0.0 %
education.num      0.0 %
marital.status     0.0 %
occupation        5.66 %
relationship       0.0 %
race               0.0 %
sex                0.0 %
capital.gain       0.0 %
capital.loss       0.0 %
hours.per.week     0.0 %
native.country    1.79 %
income             0.0 %
dtype: object
# Checking the counts of label categories
income = dataset['income'].value_counts(normalize=True)
round(income * 100, 2).astype('str') + ' %'
Output:
<=50K    75.92 %
>50K     24.08 %
Name: income, dtype: object

Observations:

1. The dataset doesn’t have any null values, but it contains missing values in the form of ‘?’ which needs to be preprocessed.
2. The dataset is unbalanced, as the dependent feature ‘income’ contains 75.92% values have income less than 50k, and 24.08% values have income more than 50k.

Step 2: Exploratory Data Analysis

2.1 Univariate Analysis:

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/059e26c9-c428-42c5-883a-53e5d7d73be5)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/be4dec45-8078-49cd-8279-429047ce1ba7)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/f2d42b8c-1cda-4f57-919f-62ad388a08c0)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/07e2eb97-1ed6-4e95-8f3b-040d7028d9cd)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/7c1ee6b1-1d0e-4e54-88c5-84932622db82)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/7f62fcf3-27ba-444a-8807-3675ac3d4340)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/cde956fe-14a7-4fd7-8b6d-1acc18722ef5)

2.2 Bivariate Analysis:

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/3deafd40-09af-4a00-a89e-58cba5272fd8)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/a85d6834-427a-4cab-8723-a82b066fb47e)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/ec796f8a-6dfb-48c7-b2e5-efa4af887486)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/5c795b11-a478-4ea2-a91c-501503e95158)

2.3 Multivariate Analysis:

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/e147caed-4dc1-4486-9007-d503df1ddfb7)
Pair plot of dataset

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/983054bb-37e4-4c8a-a1c4-f85b87228e63)
Heatmap of the correlation matrix

Observations:

1. In this dataset, the most number of people are young, white, male, high school graduates with 9 to 10 years of education, and work 40 hours per week.
2. From the correlation heatmap, we can see that the dependent feature ‘income’ is highly correlated with age, numbers of years of education, capital gain, and the number of hours per week.

Step 3: Data preprocessing

The null values are in the form of ‘?’ which can be easily replaced with the most frequent value(mode) using the fillna() command.

dataset = dataset.replace('?', np.nan)
# Checking null values
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'
Output:
age                0.0 %
workclass         5.64 %
fnlwgt             0.0 %
education          0.0 %
education.num      0.0 %
marital.status     0.0 %
occupation        5.66 %
relationship       0.0 %
race               0.0 %
sex                0.0 %
capital.gain       0.0 %
capital.loss       0.0 %
hours.per.week     0.0 %
native.country    1.79 %
income             0.0 %
dtype: object
columns_with_nan = ['workclass', 'occupation', 'native.country']
for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace = True)
    
The object columns in the dataset need to be encoded so that they can be further used. This can be done using Label Encoder in the sklearn’s preprocessing library.

from sklearn.preprocessing import LabelEncoder
for col in dataset.columns:
  if dataset[col].dtypes == 'object':         
    encoder = LabelEncoder()         
    dataset[col] = encoder.fit_transform(dataset[col])
    
The dataset is then split into X which contains all the independent features and Y which contains the dependent feature ‘Income’.

X = dataset.drop('income', axis = 1) 
Y = dataset['income']

The curse of multicollinearity and the problem of overfitting can be solved by performing Feature Selection. The feature importances can be easily found by using the ExtraTreesClassifier.

from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(random_state = 42)
selector.fit(X, Y)
feature_imp = selector.feature_importances_
for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))
Output:
0 15.59
1 4.13
2 16.71
3 3.87
4 8.66
5 8.04
6 7.27
7 8.62
8 1.47
9 2.84
10 8.83
11 2.81
12 9.64
13 1.53
X = X.drop(['workclass', 'education', 'race', 'sex', 'capital.loss', 'native.country'], axis = 1)

Using Feature Scaling we can standardize the dataset to help the model learn the patterns. This can be done with StandardScaler() from sklearn’s preprocessing library.

from sklearn.preprocessing import StandardScaler
for col in X.columns:     
  scaler = StandardScaler()     
  X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
  
The dependent feature ‘Income’ is highly imbalanced as 75.92% values have income less than 50k and 24.08% values have income more than 50k. This needs to be fixed as it results in a low F1 score. As we have a small dataset we can perform Oversampling using a technique like RandomOverSampler.

round(Y.value_counts(normalize=True) * 100, 2).astype('str') + ' %'
Output:
0    75.92 %
1    24.08 %
Name: income, dtype: object
from imblearn.over_sampling import RandomOverSampler 
ros = RandomOverSampler(random_state = 42)
ros.fit(X, Y)
X_resampled, Y_resampled = ros.fit_resample(X, Y)
round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'
Output:
1    50.0 %
0    50.0 %
Name: income, dtype: object

The dataset is split into training data and testing data in the ratio 80:20 using the train_test_split() command.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.2, random_state = 42)
print("X_train shape:", X_train.shape) 
print("X_test shape:", X_test.shape) 
print("Y_train shape:", Y_train.shape) 
print("Y_test shape:", Y_test.shape)
Output:
X_train shape: (39552, 8)
X_test shape: (9888, 8)
Y_train shape: (39552,)
Y_test shape: (9888,)

Step 4: Data Modelling

Random Forest Classifier:

from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state = 42)
ran_for.fit(X_train, Y_train)
Y_pred_ran_for = ran_for.predict(X_test)

Understanding the Algorithm:
     Random forest is a Supervised learning algorithm that is used for both classification and regression. It is a type of bagging ensemble algorithm, which creates multiple decision trees simultaneously trying to learn from the dataset independent of one another. The final prediction is selected using majority voting.
 
![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/219c2ed3-ac44-47bc-a408-77dc4eb36215)
 
Random forests are very flexible and give high accuracy as it overcomes the problem of overfitting by combining the results of multiple decision trees. Even for large datasets, random forests give a good performance. They also give good accuracy if our dataset has a large number of missing values. But random forests are more complex and computationally intensive than decision trees resulting in a time-consuming model building process. They are also harder to interpret and less intuitive than a decision tree.

This algorithm has some important parameters like max_depth, max_features, n_estimators, and min_sample_leaf. The number of trees which can be used to build the model is defined by n_estimators. Max_features determines the maximum number of features the random forest can use in an individual tree. The maximum depth of the decision trees is given by the parameter max_depth. The minimum number of samples required at a leaf node is given by min_sample_leaf.

Step 5: Model Evaluation

In this step, we will evaluate our model using two metrics which are accuracy_score and f1_score. Accuracy is the ratio of correct predicted values over the total predicted values. It tells us how accurate our prediction is. F1 score is the weighted average of precision and recall and higher its value better the model. We will use the accuracy score with f1 score as we have an imbalanced dataset.

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print('Random Forest Classifier:')
print('Accuracy score:',round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:',round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))
Output:
Random Forest Classifier:
Accuracy score: 92.6
F1 score: 92.93

Step 6: Hyperparameter Tuning

We will tune the hyperparameters of our random forest classifier using RandomizedSearchCV which finds the best hyperparameters by searching randomly avoiding unnecessary computation. We will try to find the best values for ‘n_estimators’ and ‘max_depth’.

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 40, stop = 150, num = 15)]
max_depth = [int(x) for x in np.linspace(40, 150, num = 15)]
param_dist = {
    'n_estimators' : n_estimators,
    'max_depth' : max_depth,
}
rf_tuned = RandomForestClassifier(random_state = 42)
rf_cv = RandomizedSearchCV(estimator = rf_tuned, param_distributions = param_dist, cv = 5, random_state = 42)
rf_cv.fit(X_train, Y_train)
rf_cv.best_score_
Output:
0.9131271105332539
rf_cv.best_params_
Output:
{'n_estimators': 40, 'max_depth': 102}
rf_best = RandomForestClassifier(max_depth = 102, n_estimators = 40, random_state = 42)
rf_best.fit(X_train, Y_train)
Y_pred_rf_best = rf_best.predict(X_test)
print('Random Forest Classifier:') 
print('Accuracy score:',round(accuracy_score(Y_test, Y_pred_rf_best) * 100, 2)) 
print('F1 score:',round(f1_score(Y_test, Y_pred_rf_best) * 100, 2))
Output:
Random Forest Classifier:
Accuracy score: 92.77
F1 score: 93.08
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test, Y_pred_rf_best)

![image](https://github.com/monkey-d-luffy1/Salary-prediction-in-machine-learning-using-random-forest/assets/88392078/8d34c123-fa60-4ac3-9002-fae0aae2fe0a)
Heatmap of the confusion matrix

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_rf_best))
Output:
               precision    recall   f1-score  support

           0       0.97      0.88      0.92      4955
           1       0.89      0.98      0.93      4933

    accuracy                           0.93      9888
   macro avg       0.93      0.93      0.93      9888
weighted avg       0.93      0.93      0.93      9888

The model gives us the best values for an accuracy score of 92.77 and f1 score of 93.08 after tuning its hyperparameters.

SUMMARY:

I used the Random Forest Classifier model which gives 93% accuracy on the testing data. This is a simple Random Forest Classifier model which can be easily understandable by beginners.

CONCLUSION:

In this paper we proposed a salary prediction system by using a random forest algorithm. For the proper salary prediction, we found out the most relevant feature. The result of the system is calculated by a suitable algorithm by comparing it with other algorithms in terms of standard scores and curves like the classification accuracy, the score, the ROC curve. We compared the algorithm with the basic model. Moreover, we continued with the basic model and found out the most appropriate method to add more attributes and with the highest accuracy of 93%.

Future work:

We have a large enough dataset, so we can use neural networks such as an artificial neural network to build a model that can result in better performance.
