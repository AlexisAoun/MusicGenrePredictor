import numpy as np 
import pandas as pd
import sklearn 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV


ds_test = pd.read_csv("data/spotify_dataset_test.csv")
ds_train = pd.read_csv("data/spotify_dataset_train.csv")

y_train = ds_train.genre
x_train = ds_train[ds_train.columns[:-1]]
x_train = x_train.drop("release_date",axis=1)
x_train = x_train.drop("explicit",axis=1)
x_train = x_train.drop("duration_ms",axis=1)

XTrain,XTest,yTrain,yTest = model_selection.train_test_split(x_train,y_train,test_size=500,random_state=1)

''' DecisionTreeClassifier '''

arbre1 = DecisionTreeClassifier(random_state=0)

arbre1.fit(XTrain,yTrain)

#plt.figure(figsize=(10,10))
#plot_tree(arbre1, feature_names=x_train.columns, filled= True)
#plt.show()
pred_tree = arbre1.predict(XTest)
print(accuracy_score(pred_tree,yTest))


'''Random Forest algo'''

#RAndom Forest 



param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv= 5)
CV_rfc.fit(XTrain, yTrain)

CV_rfc.best_params_


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
rfc1.fit(x_train, y_train)
pred=rfc1.predict(XTest)
print("Accuracy for Random Forest on CV data: ",accuracy_score(pred,yTest))
