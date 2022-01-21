import numpy as np 
import pandas as pd
import sklearn
from sklearn import model_selection
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from sklearn import metrics




ds_test = pd.read_csv("data/spotify_dataset_test.csv")
ds_train = pd.read_csv("data/spotify_dataset_train.csv")

y_train = ds_train.genre
X_train = ds_train[ds_train.columns[:-1]]
X_train = X_train.drop("release_date",axis=1)
X_train = X_train.drop("explicit",axis=1)
X_train = X_train.drop("duration_ms",axis=1)




XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X_train,y_train,test_size=500,random_state=1,stratify= y_train)

x_val = XTrain[-10000:]
y_val = y_train[-10000:]
x_train = XTrain[:-10000]
y_train = y_train[:-10000]

#instanciation du modèle


modelSimple = Sequential()

modelSimple.add(tf.keras.layers.Dense(256, input_shape=(XTrain.shape[1],), activation='sigmoid'))

print(type(modelSimple))



modelSimple.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])

modelSimple = modelSimple.fit(XTrain,yTrain,epochs=150,batch_size=32, validation_data=(x_val, y_val),
)


print(modelSimple.get_weights())

predSimple = modelSimple.predict_classes(XTest)

print(predSimple[:10])

print(metrics.confusion_matrix(yTest,predSimple))
print(metrics.accuracy_score(yTest,predSimple))

#outil dédié
score = modelSimple.evaluate(XTest,yTest)
print(score)