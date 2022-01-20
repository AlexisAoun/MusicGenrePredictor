# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:16:05 2022

@author: aberg

Random Forest
Naive Bayes
Stochastic Gradient Descent Classifier
Logistic Regression

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


ds_test=pd.read_csv("data/spotify_dataset_test.csv")
ds_train=pd.read_csv("data/spotify_dataset_train.csv")

print("Taille du dataset_test :",np.shape( ds_test))
print("Taille du dataset_train :",np.shape(ds_train))

ds_test.info()
ds_train.info()

y=ds_train.iloc[0:500000, [16]]
ds_test=ds_test.drop(columns = ['release_date','explicit','mode','key'])
ds_train=ds_train.drop(columns = ['release_date','explicit','mode','key','genre'])


#1er essais avec un train_test_split et quelques classifieurs
X_train, X_test, y_train, y_test = train_test_split(ds_train,y)
#clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,random_state=0)
clf = KNeighborsClassifier(n_neighbors=1000)
#clf=LinearDiscriminantAnalysis()
#clf = QuadraticDiscriminantAnalysis()
#clf = LogisticRegression(random_state=0)
#clf= GaussianNB()
clf.fit(X_train, y_train)
y_predict=clf.predict(X_test)








print("-Accuracy : ", accuracy_score(y_test, y_predict))


'''
#kfolds
scores = []
clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
cv = KFold(n_splits=10)
for train_index, test_index in cv.split(ds_train) :
    print(train_index)
    print(ds_train)
    X_train=ds_train.iloc[train_index]
    y_train=y.iloc[train_index]
    X_test=ds_train.iloc[test_index]
    y_test=y.iloc[test_index]
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

print("results",scores)

'''
    



