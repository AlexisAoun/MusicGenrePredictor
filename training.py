import model 
import datasetPrep

m = model.getModel() 
trainX, trainY, testX, testY = datasetPrep.computeDatasets()

def train(epochs=50):
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(trainX, trainY, epochs=epochs)
    model.save('models/modelTest')

train()
score = m.evaluate(testX, testY, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
