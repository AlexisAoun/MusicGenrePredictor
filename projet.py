# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:16:05 2022

@author: aberg
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ds_test=pd.read_csv("data/spotify_dataset_test.csv")
ds_train=pd.read_csv("data/spotify_dataset_train.csv")

print("Taille du dataset_test :",np.shape( ds_test))
print("Taille du dataset_train :",np.shape(ds_train))

ds_test.info()
ds_train.info()

y=ds_train.iloc[0:500000, [16]]
ds_test=ds_test.drop(columns = ['release_date','explicit','mode','key'])
ds_train=ds_train.drop(columns = ['release_date','explicit','mode','key','genre'])


#Essai avec le randomforest
clf = RandomForestClassifier()
clf.fit(ds_train, y)

print(clf.predict(ds_test))



