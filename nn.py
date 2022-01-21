import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from random import randrange

#----------data extraction/preparing the training and testing sets---------------------
data=pd.read_csv("data/spotify_dataset_train.csv")


sizeOfTestingSet = 300

labels = data['genre']
data = data.drop(columns='genre')

data = data.to_numpy()
labels = labels.to_numpy()

genres = ['r&b' ,'rap' ,'classical' ,'salsa' ,'edm' ,'hip hop' ,'trap' ,'techno' 
,'jazz' ,'metal' ,'country' ,'rock' ,'reggae' , 'latin' ,'disco' ,'soul' ,'chanson' 
,'blues' ,'dance' ,'electro' ,'punk' ,'folk' ,'pop']
#-------------------1-normalize the features 
#features to be normalized : popularity, key, loundess, tempo, duration_ms, time_signature

#convert all dates into years (and to int in the process)
#change explicit values from boolean to int (0,1)
for row in data:
    if len(row[0]) > 7:  
        row[0] = datetime.strptime(row[0], "%Y-%m-%d").year
    elif len(row[0]) > 4:
        row[0] = datetime.strptime(row[0], "%Y-%m").year
    else:
        row[0] = datetime.strptime(row[0], "%Y").year
    
    if row[1]:
        row[1] = 1
    else:
        row[1] = 0

#normalizing the values to (0,1)
scalledData = np.empty(shape=(data.shape))
for i in range(16):
   scaler = MinMaxScaler(feature_range=(0,1))
   column = data[:,i].reshape(-1,1)
   scaler.fit(column)
   scalledData[:,i] = scaler.transform(column)[:,0]

#replacing genres name by there respective indexes 
labelsInt = []
for row in labels:
    labelsInt.append(genres.index(row))

labelsInt = np.array(labelsInt)
#-----------------2-splitting the dataset between training and testing set
#warning: not the most optimized way (at all), could hurt your eyes

testX = []
testY = []

trainX = []
trainY = []

picked = []
for i in range(sizeOfTestingSet):
    tmp = randrange(scalledData.shape[0])
    if tmp not in picked:
       picked.append(tmp)  

print(picked)
c=0
for i in scalledData:
    if c in picked:
        testX.append(i)
        testY.append(labelsInt[c])
    else:
        trainX.append(i)
        trainY.append(labelsInt[c])
    c+=1

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

##--------------The Model --------------------------------------
#
#model = tf.keras.Sequential(
#    [ 
#        tf.keras.Input(shape=(16)),
#        tf.keras.layers.Dense(256, activation=tf.nn.relu),
#        tf.keras.layers.Dense(128, activation=tf.nn.relu),
#        tf.keras.layers.Dense(23, activation=tf.nn.softmax)
#    ]
#)
#
#model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#model.fit(trainX, trainY, epochs=50)
#
##model.save('models/model1')
#
##model = tf.keras.models.load_model('models/model1')
#
#score = model.evaluate(testX, testY, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])

