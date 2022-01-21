import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn 
from sklearn.manifold import TSNE


ds_test = pd.read_csv("data/spotify_dataset_test.csv")
ds_train = pd.read_csv("data/spotify_dataset_train.csv")

y_train = ds_train.genre
x_train = ds_train[ds_train.columns[:-1]]
x_train = x_train.drop("release_date",axis=1)
x_train = x_train.drop("explicit",axis=1)
x_train = x_train.drop("duration_ms",axis=1)

X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit(x_train,y_train)
