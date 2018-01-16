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
from sklearn.ensemble import RandomForestRegressor
import pickle


#inputFile = loadtxt("planVectorsSGD2-kmeans-simword-opportuneWordcount.txt", comments="#", delimiter=" ", unpack=False)
inputFile = loadtxt("mlModelVectors.txt", comments="#", delimiter=" ", unpack=False)

#size = 146;
#start = 13;
size = 213
start = 5
x_test = inputFile[start:,0:size]
y_test = inputFile[start:,size]

# x_train = inputFile[:,0:size]
# y_train = inputFile[:,size]
#
# x_test = inputFile[:,0:size]
# y_test = inputFile[:,size]



# load the model from disk
filename = 'ForestModel.sav'
regr = pickle.load(open(filename, 'rb'))



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(regr, x_train, y_train, cv=kfold)
#accuracy_score(prediction,y_train)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
prediction = regr.predict(x_test)

for num in range(1, 34):
    if num % 2 == 0:
        print("estimated time for " + str(x_test[num][size-2]) + "-" + str(x_test[num][size-1]) + " in java : " + str(
            prediction[num]) + "(real " + str(y_test[num]) + ")")
    else:
        print("estimated time for " + str(x_test[num][size-2]) + "-" + str(x_test[num][size-1]) + " in spark : " + str(
            prediction[num]) + "(real " + str(y_test[num]) + ")")

# print results to text
text_file = open("estimates.txt", "w")
for num in range(1, prediction.size):
    text_file.write("%d" % prediction[num])
    text_file.write("\n")
text_file.close()

