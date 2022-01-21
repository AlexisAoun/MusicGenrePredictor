
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score





ds_test = pd.read_csv("data/spotify_dataset_test.csv")
ds_train = pd.read_csv("data/spotify_dataset_train.csv")

y_train = ds_train.genre
x_train = ds_train[ds_train.columns[:-1]]
x_train = x_train.drop("release_date",axis=1)
x_train = x_train.drop("explicit",axis=1)
x_train = x_train.drop("duration_ms",axis=1)

XTrain,XTest,yTrain,yTest = model_selection.train_test_split(x_train,y_train,test_size=500,random_state=1)


param_grid = {'n_neighbors' : np.arange(44,60), 'metric' : ['euclidean', 'manhattan']}

grid = GridSearchCV(KNeighborsClassifier(),param_grid, cv=5 ) # cross validation =5

grid.fit(XTrain, yTrain)

print(grid.best_score_) 
print(grid.best_estimator_)
print(grid.best_params_)
"""
"""
# Résultats  : 0.3675272445900707   KNeighborsClassifier(metric='manhattan', n_neighbors=19)   {'metric': 'manhattan', 'n_neighbors': 19}

# ESSai avec les paramètres fournis par GridSearch 

KNNclassifier = KNeighborsClassifier(metric='manhattan', n_neighbors=44) 
KNNclassifier=KNNclassifier.fit(XTrain, yTrain)


# prediction

pred = KNNclassifier.predict(XTest)

print("Accuracy for KNN on CV data: ",accuracy_score(yTest,pred))