from pathlib import Path

import numpy
import sys

from keras.layers import Dense
from keras.models import load_model, Sequential
from numpy import loadtxt
import pickle
import os

from sklearn.externals import joblib

featureVectorSize = 251

def wider_deep_model():
	# create model
	model = Sequential()
	model.add(Dense(featureVectorSize+20, input_dim=featureVectorSize, kernel_initializer='normal', activation='relu'))
	model.add(Dense(55, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def main():

    #inputFile = loadtxt("planVectorsSGD2-kmeans-simword-opportuneWordcount.txt", comments="#", delimiter=" ", unpack=False)

    currentDirPath =os.path.dirname(os.path.realpath(__file__))

    dirPath = str(Path.home())

    model="nn"
    if (len(sys.argv)>=2):
        model = sys.argv[1]


    if (len(sys.argv)>=3):
        inputFile = loadtxt(sys.argv[2], comments="#", delimiter=" ", unpack=False)
    else:
        inputFile = loadtxt(os.path.join(dirPath,".rheem","mlModelVectors.txt"), comments="#", delimiter=" ", unpack=False)

    #size = 146;
    #start = 13;
    #size = 213
    size=251
    start = 0
    dimInputFile = inputFile.ndim

    if(dimInputFile==1):
        inputFile = numpy.reshape(inputFile, (-1,inputFile.size))
    x_test = inputFile[:,0:size]
    y_test = inputFile[start:,size]

    # x_train = inputFile[:,0:size]
    # y_train = inputFile[:,size]
    #
    # x_test = inputFile[:,0:size]
    # y_test = inputFile[:,size]



    # load the model from disk
    if(model=="forest"):
        # load the model from disk
        filename = os.path.join(currentDirPath, "ForestModel.sav")
        print("Loading model: "+filename)
        model = pickle.load(open(filename, 'rb'))
    elif(model=="nn"):
        filename = os.path.join(currentDirPath,'nn.pkl')
        print("Loading model: "+filename)
        # Load the pipeline first:
        model = joblib.load(filename)

        # Then, load the Keras model:
        model.named_steps['mlp'].model = load_model(os.path.join(currentDirPath,'keras_model.h5'))




    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    #kfold = KFold(n_splits=10, random_state=seed)
    #results = cross_val_score(regr, x_train, y_train, cv=kfold)
    #accuracy_score(prediction,y_train)
    #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    prediction = model.predict(x_test)

    # for num in range(1,min([34,len(x_test)])):
    #     if num % 2 == 0:
    #         print("estimated time for " + str(x_test[num][size-2]) + "-" + str(x_test[num][size-1]) + " in java : " + str(
    #             prediction[num]) + "(real " + str(y_test[num]) + ")")
    #     else:
    #         print("estimated time for " + str(x_test[num][size-2]) + "-" + str(x_test[num][size-1]) + " in spark : " + str(
    #             prediction[num]) + "(real " + str(y_test[num]) + ")")

    # print results to text
    if (len(sys.argv) >= 4):
        saveLocation = loadtxt(sys.argv[3], comments="#", delimiter=" ", unpack=False)
    else:
        saveLocation = os.path.join(dirPath, ".rheem", "estimates.txt")

    # delete first
    if(os._exists(saveLocation)):
        os.remove(saveLocation)
    text_file = open(saveLocation, "w")

    # print estimates
    dimResults = prediction.ndim
    if (dimResults == 0):
        text_file.write("%d" % prediction)
        text_file.write("\n")
    else:
        for num in range(0, prediction.size):
            t = prediction[num]
            text_file.write("%d" % prediction[num])
            text_file.write("\n")
    text_file.close()
    print("estimation done!")

if __name__ == "__main__":
   main()
