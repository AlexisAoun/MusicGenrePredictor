import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.svm import SVC



ds_test = pd.read_csv("data/spotify_dataset_test.csv")
ds_train = pd.read_csv("data/spotify_dataset_train.csv")

y_train = ds_train.genre
x_train = ds_train[ds_train.columns[:-1]]
x_train = x_train.drop("release_date",axis=1)
x_train = x_train.drop("explicit",axis=1)
x_train = x_train.drop("duration_ms",axis=1)

XTrain,XTest,yTrain,yTest = model_selection.train_test_split(x_train,y_train,test_size=500,random_state=1)

param_grid = {'C':[1,10], 'gamma':[0.1,0.01], "kernel": ["rbf"]}
  
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
grid.fit(x_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
print(grid.best_score_) 