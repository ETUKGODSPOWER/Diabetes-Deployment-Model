import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

diabetes = pd.read_csv('diabetes.csv')
diabetes.head()

diabetes["Glucose"] = diabetes["Glucose"].apply(lambda x: np.nan if x == 0 else x)
diabetes["BloodPressure"] = diabetes["BloodPressure"].apply(lambda x: np.nan if x == 0 else x)
diabetes["SkinThickness"] = diabetes["SkinThickness"].apply(lambda x: np.nan if x == 0 else x)
diabetes["Insulin"] = diabetes["Insulin"].apply(lambda x: np.nan if x == 0 else x)
diabetes["BMI"] = diabetes["BMI"].apply(lambda x: np.nan if x == 0 else x)


diabetes.skew()
diabetes['BMI'].fillna(diabetes['BMI'].median(), inplace = True)
diabetes['Glucose'].fillna(diabetes['Glucose'].median(), inplace = True)
diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean(), inplace = True)
diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].median(), inplace = True)
diabetes['Insulin'].fillna(diabetes['Insulin'].median(), inplace = True)

X = diabetes.drop(columns = 'Outcome')
y = diabetes['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

acc_baseline = y_train.value_counts(normalize = True).max()
print('Baseline Accuracy :', round(acc_baseline, 2))

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
ACC = accuracy_score(y_pred, y_test)
print(ACC)

CCC = confusion_matrix(y_pred, y_test)
print(CCC)

CF = CCC.transpose()

import pickle

with open('Diabeties_Deployment', 'wb') as file:
    pickle.dump(model, file)