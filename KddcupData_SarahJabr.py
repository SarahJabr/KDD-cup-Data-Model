# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:55:19 2022

@author: ahmad
"""
import pandas as pd
import numpy as np

data=pd.read_csv('data.csv',on_bad_lines="warn",names=[i for i in range(42)],
                 engine='python',chunksize=10000)
dataset=pd.concat(data)

print(dataset.isnull().sum(0))
print(dataset.info())
print(dataset.nunique())

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)

X1=pd.get_dummies(X[:,1],drop_first=True)
X2=pd.get_dummies(X[:,2],drop_first=True)
X3=pd.get_dummies(X[:,3],drop_first=True)

X=np.delete(X,[1,2,3], axis=1)

X=np.append(X,X1,axis=1)
X=np.append(X,X2,axis=1)
X=np.append(X,X3,axis=1)

# da=pd.get_dummies(da,columns=[1,2,3]).values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4,
                                               random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)#you use fit only once for each scaler
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics as ms
print('Acuracy=',ms.accuracy_score(y_test, y_pred))
print("Mean absolute error =", round(ms.mean_absolute_error(y_test,y_pred), 3)) 
print("Mean squared error =", round(ms.mean_squared_error(y_test,y_pred), 3)) 
print("Median absolute error =", round(ms.median_absolute_error(y_test,y_pred), 3)) 
print("Explain variance score =", round(ms.explained_variance_score(y_test,y_pred), 3))  
print("R2 score =", round(ms.r2_score(y_test, y_pred), 3))
