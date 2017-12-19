import keras
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from numpy import loadtxt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model

inputFile = loadtxt("planVectors", comments="#", delimiter=" ", unpack=False)

x_train = inputFile[:,0:105]
y_train = inputFile[:,105]

clf = linear_model.Lasso(alpha=0.1)
clf.fit(x_train,y_train)
print(clf.coef_)
print(clf.intercept_)
score = clf.score(x_train,y_train)
print(score)
prediction = clf.predict(x_train)

for num in range(1, 17):
    if num % 2 == 0:
        print("estimated time for " + str(x_train[num][103]) + "-" + str(x_train[num][104]) + " in java : " + str(
            prediction[num]) + "(real " + str(y_train[num]) + ")")
    else:
        print("estimated time for " + str(x_train[num][103]) + "-" + str(x_train[num][104]) + " in spark : " + str(
            prediction[num]) + "(real " + str(y_train[num]) + ")")

print(cross_val_score(clf, x_train, y_train))