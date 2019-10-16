# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
from sklearn import neighbors
from sklearn import svm


#reading dataset
dataset = pd.read_csv('Churn.csv')

#load
dataset.shape

#make a copy of original data
df=dataset
df.head()

#dropping cutomer ID column
df=df.drop(['customerID'],axis=1)

df.groupby(["SeniorCitizen","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))
df.groupby(["Partner","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))
df.groupby(["Dependents","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))
df.groupby(["tenure","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(15,5))
#converting string values to float type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
#filling missing values with mean of TotalCharges
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

df['Churn'].dtype
df=df.drop(['Dependents'],axis=1)
 


df.groupby(["gender","churn"])



#creating X and y
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#Label Encoding and One Hot Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
enY = LabelEncoder()
y = enY.fit_transform(y)
enX = LabelEncoder()
X[:,0] = enX.fit_transform(X[:,0])

X[:,1] = enX.fit_transform(X[:,1])

X[:,2] = enX.fit_transform(X[:,2])


X[:,4] = enX.fit_transform(X[:,4])

X[:,5] = enX.fit_transform(X[:,5])

X[:,6] = enX.fit_transform(X[:,6])

X[:,7] = enX.fit_transform(X[:,7])

X[:,8] = enX.fit_transform(X[:,8])

X[:,9] = enX.fit_transform(X[:,9])

X[:,10] = enX.fit_transform(X[:,10])

X[:,11] = enX.fit_transform(X[:,11])

X[:,14] = enX.fit_transform(X[:,14])

X[:,13] = enX.fit_transform(X[:,13])
X[:,12] = enX.fit_transform(X[:,12])
X[:,15] = enX.fit_transform(X[:,15])

X[:,16] = enX.fit_transform(X[:,16])
X[:,17] = enX.fit_transform(X[:,17])
#16 17
oneencX = OneHotEncoder(categorical_features=[5,6,7,8,9,10,11,12,13,15])
X = oneencX.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Finding Best model out of following


clf = ensemble.GradientBoostingClassifier(learning_rate=0.25, n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

metrics.confusion_matrix(y_test,y_pred)

score = metrics.accuracy_score(y_test, y_pred)





