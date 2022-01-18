# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:16:05 2022

@author: aberg
"""

import numpy as np
import pandas as pd

ds_test=pd.read_csv("data/spotify_dataset_test.csv")
ds_train=pd.read_csv("data/spotify_dataset_train.csv")
ds_subset=pd.read_csv("data/spotify_dataset_subset.csv")

print("Taille du dataset_test :",np.shape( ds_test))
print("Taille du dataset_train :",np.shape(ds_train))
#on a une colonne en plus correspondant au genre 
print("Taille du dataset dataset_subset :",np.shape(ds_subset))


ds_test.info()
ds_train.info()
ds_subset.info()


print(ds_test.head())