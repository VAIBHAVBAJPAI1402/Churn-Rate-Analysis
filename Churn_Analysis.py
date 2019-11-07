# import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
from sklearn import neighbors
from sklearn import svm

os.chdir("C:\\Users\\Vaibhav\\Desktop\\Task")
#reading dataset
dataset = pd.read_csv("Churn.csv")

#load
dataset.shape

#make a copy of original data
df=dataset
df.head()

#Replacing No internet service with No

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    df[i]  = df[i].replace({'No internet service' : 'No'})
    
df.head(10)
#Replacing 'No phone service' with 'No'

df['MultipleLines'] = df['MultipleLines'].replace({'No phone service' : 'No'})
df.head(10)
#converting tenure into categorical form

def tenure_to_category(df):
    if(int(df["tenure"])<=12):
        return "0-12"
    elif(df['tenure']<=24):
        return '12-24'
    elif(df['tenure']<=48):
        return '12-48'
    elif(df['tenure']<=60):
        return '48-60'
    elif(df['tenure']>60):
        return 'mt_60'
df['tenure_category'] = df.apply(lambda df:tenure_to_category(df),axis=1)
#df['tenure'].dtype
df.head(10)

def encode_gender(df):
    if(df['gender']=='Female'):
        return 0
    else:
        return 1

df['gender'] = df.apply(lambda df:encode_gender(df),axis=1)

#dropping cutomer ID column
df=df.drop(['customerID'],axis=1)
#dropping column tenure as we have already created a categorical form of it
df=df.drop(['tenure'],axis=1)

df.head()
#0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16
df.groupby(['tenure_category','Churn']).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))
df.groupby(["SeniorCitizen","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))
df.groupby(["Partner","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))
df.groupby(["Dependents","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))

#Significant field
df.groupby(["tenure_category","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))

df.groupby(["Contract","Churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(5,5))

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
#filling missing values with mean of TotalCharges
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

df['Churn'].dtype
X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19]].values
y = df.iloc[:,18].values
print(X)
print(y)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
enY = LabelEncoder()
y = enY.fit_transform(y)

enX = LabelEncoder()
X[:,0] = enX.fit_transform(X[:,0])

X[:,1] = enX.fit_transform(X[:,1])

X[:,2] = enX.fit_transform(X[:,2])

X[:,3] = enX.fit_transform(X[:,3])

X[:,4] = enX.fit_transform(X[:,4])

X[:,5] = enX.fit_transform(X[:,5])

X[:,6] = enX.fit_transform(X[:,6])

X[:,7] = enX.fit_transform(X[:,7])

X[:,8] = enX.fit_transform(X[:,8])

X[:,9] = enX.fit_transform(X[:,9])

X[:,10] = enX.fit_transform(X[:,10])

X[:,11] = enX.fit_transform(X[:,11])

X[:,12] = enX.fit_transform(X[:,12])

X[:,14] = enX.fit_transform(X[:,14])

X[:,13] = enX.fit_transform(X[:,13])

X[:,15] = enX.fit_transform(X[:,15])

X[:,18] = enX.fit_transform(X[:,18])
#0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16
oneencX = OneHotEncoder(categorical_features=[6,13,15,18])
X = oneencX.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
print(y_train)

clf = ensemble.GradientBoostingClassifier(learning_rate=0.25, n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(y_pred)

import pylab as pl
cm=metrics.confusion_matrix(y_test,y_pred)
pl.matshow(cm)
pl.title('Confusion Matrix')
pl.colorbar()
pl.show()

score = metrics.accuracy_score(y_test, y_pred)
print(score)