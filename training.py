import model 
import datasetPrep

m = model.getModel() 

def train(epochs=50):
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    trainX, trainY, testX, testY = datasetPrep.computeDatasets()
    model.fit(trainX, trainY, epochs=epochs)
    model.save('models/modelTest')

train()
