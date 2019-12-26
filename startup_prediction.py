# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:37:50 2019

@author: Sabitha Jaleel
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


dataset.head(6)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

# Accuracy test
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

df =pd.DataFrame(data=y_test,columns=['y_test'])
df['y_pred'] = y_pred

import pickle
pickle.dump(regressor,open('Startup_prediction.pkl','wb'))

regressor.predict([[1,0,0,165349.20,136897.80,471784.10]])

X_test
