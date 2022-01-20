import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

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
#warning: not optimized at all, could hurt your eyes

testX = []
testY = []

trainX = []
trainY = []

c = 0
for i in scalledData:
    if c < sizeOfTestingSet:
        trainX.append(i)
        trainY.append(labelsInt[c])
    else:
        testX.append(i)
        testY.append(labelsInt[c])
    c+=1

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

print(scalledData.shape)
print(testX.shape)
print(testY.shape)
print(trainX.shape)
print(trainY.shape)
