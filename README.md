# Prediction_Heart_Failure_Using_RF

## Business Problem
In this section we are planing to predict the Heart Failure by spesific parameters and refered parameters described as.

## About This Dataset
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
## Import Necessary Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

import warnings
warnings.filterwarnings('ignore')
```

## General Information About the Dataset
### Checking the Data Frame
Since we want to check the data to get a general idea about it, we create and use a function called check_df(dataframe, head=5, tail=5) that prints the referenced functions:
```
print(20*"#","HEAD",20*"#")
print(dataframe.head(head))
print(20*"#","Tail",20*"#")
print(dataframe.tail(head))
print(20*"#","Shape",20*"#")
print(dataframe.shape)
print(20*"#","Types",20*"#")
print(dataframe.dtypes)
print(20*"#","NA",20*"#")
print(dataframe.isnull().sum().sum())
print(dataframe.isnull().sum())
print(20*"#","Quartiles",20*"#")
print(dataframe.describe([0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]).T
```

## Analysis of Categorical and Numerical Variables
```
cat_cols = [col for col in datraframe.columns if str(datraframe[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in datraframe.columns if datraframe[col].nunique()< cat_th and datraframe[col].dtypes in ["uint8", "int64", "float64"]]
cat_but_car = [col for col in datraframe.columns if datraframe[col].nunique() > car_th and str(datraframe[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
num_cols= [col for col in datraframe.columns if datraframe[col].dtypes in ["uint8", "int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]
```
We create another plot function called plot_num_summary(dataframe) to see the whole summary of numerical columns due to the high quantity of them:
![plot_num](https://github.com/user-attachments/assets/049d2bb2-a302-47bf-af62-2d17fb3fd716)

## Correlation analysis
To analyze correlations between numerical columns we create a function called correlated_cols(dataframe):
![Correlation analysis](https://github.com/user-attachments/assets/ed069d22-ec18-4a3c-9135-20a25cc2f8cb)

## Missing Value
We check the data to designate the missing values in it, dataframe.isnull().sum():
* age                         0
* anaemia                     0
* creatinine_phosphokinase    0
* diabetes                    0
* ejection_fraction           0
* high_blood_pressure         0
* platelets                   0
* serum_creatinine            0
* serum_sodium                0
* sex                         0
* smoking                     0
* time                        0
* DEATH_EVENT                 0
dtype: int64

## Random Forest
We create our model and see the results:

#################### Accuracy & Results ####################

Accuracy Train :  1.000

Accuracy Test :  0.767

R2 Train :  1.000

R2 Test :  0.767

Cross Validation Train:  0.862

Cross Validation Test:  0.744

Cross Validation (Accucary) 0.692

Cross Validation (F1) 0.456

Cross Validation (Recall) 0.537

Cross Validation (Precision) 0.656

Cross Validation (roc_auc) 0.850

![rf](https://github.com/user-attachments/assets/1c5b8cb9-2ffa-4a50-b5f2-916cebc1a510)

## Model Tuning
After creating our model, we proceed to fine-tune it and evaluate the results:

#################### Accuracy & Results ####################

Accuracy Train :  0.962

Accuracy Test :  0.878

R2 Train :  0.962

R2 Test :  0.878

Cross Validation Train:  0.885

Cross Validation Test:  0.778

Cross Validation (Accucary) 0.719

Cross Validation (F1) 0.487

Cross Validation (Recall) 0.537

Cross Validation (Precision) 0.716

Cross Validation (roc_auc) 0.879

![rf_tune](https://github.com/user-attachments/assets/d322ff49-4112-40e9-8aee-71269ca9475a)

## Loading a Base Model and Prediction
Via joblib we can save and/or load our model:
```
def load_model(pklfile):
  model_disc = joblib.load(pklfile)
  return model_disc
```
---
Now we can make predictions with our model:
```
X = df.drop("DEATH_EVENT", axis=1)
x = X.sample(1).values.tolist()
model_disc.predict(pd.DataFrame(X))[0]
```
result = 1

---
```
sample2 = [50, 1, 900, 0, 20, 0, 327000.0000, 1.9000, 140, 0, 1, 5]
model_disc.predict(pd.DataFrame(sample2).T)[0]
```
result=1
