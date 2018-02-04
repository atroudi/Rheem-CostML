import numpy
import sys
from numpy import loadtxt
import pickle
import os

def main():
    #inputFile = loadtxt("planVectorsSGD2-kmeans-simword-opportuneWordcount.txt", comments="#", delimiter=" ", unpack=False)

    dirPath =os.path.dirname(os.path.realpath(__file__))

    if (len(sys.argv)>=2):
        inputFile = loadtxt(sys.argv[1], comments="#", delimiter=" ", unpack=False)
    else:
        inputFile = loadtxt(dirPath+"\\mlModelVectors.txt", comments="#", delimiter=" ", unpack=False)

    #size = 146;
    #start = 13;
    #size = 213
    size=251
    start = 0
    x_test = inputFile[:,0:size]
    y_test = inputFile[start:,size]

    # x_train = inputFile[:,0:size]
    # y_train = inputFile[:,size]
    #
    # x_test = inputFile[:,0:size]
    # y_test = inputFile[:,size]



    # load the model from disk
    filename = dirPath+'\\ForestModel.sav'
    regr = pickle.load(open(filename, 'rb'))



    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    #kfold = KFold(n_splits=10, random_state=seed)
    #results = cross_val_score(regr, x_train, y_train, cv=kfold)
    #accuracy_score(prediction,y_train)
    #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    prediction = regr.predict(x_test)

    # for num in range(1,min([34,len(x_test)])):
    #     if num % 2 == 0:
    #         print("estimated time for " + str(x_test[num][size-2]) + "-" + str(x_test[num][size-1]) + " in java : " + str(
    #             prediction[num]) + "(real " + str(y_test[num]) + ")")
    #     else:
    #         print("estimated time for " + str(x_test[num][size-2]) + "-" + str(x_test[num][size-1]) + " in spark : " + str(
    #             prediction[num]) + "(real " + str(y_test[num]) + ")")

    # print results to text
    if (len(sys.argv) >= 3):
        saveLocation = loadtxt(sys.argv[2], comments="#", delimiter=" ", unpack=False)
    else:
        saveLocation = dirPath+"\\estimates.txt"

    # delete first
    if(os._exists(saveLocation)):
        os.remove(saveLocation)
    text_file = open(saveLocation, "w")
    for num in range(1, prediction.size):
        text_file.write("%d" % prediction[num])
        text_file.write("\n")
    text_file.close()
    print("estimation done!")

if __name__ == "__main__":
   main()
