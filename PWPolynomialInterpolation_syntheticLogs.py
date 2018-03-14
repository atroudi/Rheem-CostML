import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from numpy import loadtxt, copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn import linear_model

#inputFile = loadtxt("planVectors", comments="#", delimiter=" ", unpack=False)

def main():
    featureVectorSize = 251
    startTraining = 5
    minimumTrainingSamples=7
    interpolationDegrees=[4]
    #endTraining = 13

    RheemPath=str(Path.home())
    inputDirectoryLocation=os.path.join(RheemPath,".rheem","syntheticLog")

    # create output directory
    outputDirectoryLocation=os.path.join(inputDirectoryLocation,"output")
    outputFigureDirectoryLocation=os.path.join(inputDirectoryLocation,"output","Figures")
    if not os.path.exists(outputDirectoryLocation):
        os.makedirs(outputDirectoryLocation)

    if not os.path.exists(outputFigureDirectoryLocation):
        os.makedirs(outputFigureDirectoryLocation)

    for file in os.listdir(inputDirectoryLocation):
        #file='1P0J0L1D_Sel1_Compx1_Operators_reduce1_callbacksink1_map4_textsource1_ApacheSpark7.log'
        if file.endswith(".log"):


            print(file)
            inputFileLocation=os.path.join(inputDirectoryLocation,file)
            #inputFileLocation="resources/syntheticLogs4"
            try:
                endTraining = len(open(inputFileLocation).readlines())
            except:
                endTraining=0
            if(endTraining>startTraining+minimumTrainingSamples):
                print(endTraining)
                inputFile = loadtxt(inputFileLocation, comments="#", delimiter=" ", unpack=False)

                output_medium_file = open(os.path.join(outputDirectoryLocation,file.replace(".log","_synthetic_medium.log")), "w")
                output_high_file = open(os.path.join(outputDirectoryLocation,file.replace(".log","_synthetic_high.log")), "w")

                x_train = inputFile[startTraining:endTraining, 0:featureVectorSize]
                y_train = inputFile[startTraining:endTraining, featureVectorSize]

                lasso_model = linear_model.Lasso(alpha=0.1)
                lasso_model.fit(x_train, y_train)

                # regr = RandomForestRegressor(max_depth=25, random_state=0)
                # regr.fit(x_train,y_train)
                #
                # print(lasso_model.coef_)
                # print(lasso_model.intercept_)
                # score = lasso_model.score(x_train, y_train)
                # print(score)
                # prediction = clf.predict(x_train)
                #
                # for num in range(1, start):
                #     if num % 2 == 0:
                #         print("estimated time for " + str(x_train[num][size-2]) + "-" + str(x_train[num][size-3]) + " in java : " + str(
                #             prediction[num]) + "(real " + str(y_train[num]) + ")")
                #     else:
                #         print("estimated time for " + str(x_train[num][size-2]) + "-" + str(x_train[num][size-3]) + " in spark : " + str(
                #             prediction[num]) + "(real " + str(y_train[num]) + ")")
                #
                # print(cross_val_score(clf, x_train, y_train))

                # Handle testing
                # Generate testing vectors
                step = 100000
                samples = 200
                startIQ = 10

                # Log vector parameters
                operatorsOffset=4
                numberOperators=20
                operatorFeatures=10
                inputCardinalityOffset=8
                outputCardinalityOffset=9

                x_test = np.empty((samples,featureVectorSize))
                y_test = np.empty(samples)

                # Get the first synthetic original log
                synthetic_original_log = x_train[0]

                # Create a synthetic test vector with multipying the io cardinalities for each duplicate
                for sample in range(0, samples):
                    newSyntheticLog = copy(synthetic_original_log)
                    for operatorPos in range(0, numberOperators):
                        if(synthetic_original_log[operatorsOffset+operatorPos*operatorFeatures+inputCardinalityOffset]!=0):
                            selectivity = synthetic_original_log[operatorsOffset+operatorPos*operatorFeatures+outputCardinalityOffset]/ \
                                        synthetic_original_log[operatorsOffset+operatorPos*operatorFeatures+inputCardinalityOffset]
                            newSyntheticLog[operatorsOffset + operatorPos * operatorFeatures + inputCardinalityOffset] = \
                                synthetic_original_log[operatorsOffset + operatorPos * operatorFeatures + inputCardinalityOffset] + sample * step * selectivity
                            newSyntheticLog[operatorsOffset + operatorPos * operatorFeatures + outputCardinalityOffset] = \
                                synthetic_original_log[operatorsOffset + operatorPos * operatorFeatures + outputCardinalityOffset] + sample * step * selectivity

                    # Update input cardinality
                    newSyntheticLog[featureVectorSize - 2] = \
                        int(synthetic_original_log[featureVectorSize - 2] + sample * step)
                        # newSyntheticLog[operatorsOffset+operatorPos*operatorFeatures+inputCardinalityOffset]= \
                        #     synthetic_original_log[operatorsOffset+operatorPos*operatorFeatures+inputCardinalityOffset]*(sample+1)
                        # newSyntheticLog[operatorsOffset+operatorPos*operatorFeatures+outputCardinalityOffset]= \
                        #     synthetic_original_log[operatorsOffset+operatorPos*operatorFeatures+outputCardinalityOffset]*(sample+1)
                    # Update input cardinality
                    # newSyntheticLog[size-2] = \
                    # synthetic_original_log[size-2] * (sample + 1)
                    x_test[sample]=newSyntheticLog
                    #np.insert(x_test,sample,newSyntheticLog)
                    #synthetic_original_log = x_train[startIQ]

                # print cross validation
                # prediction = lasso_model.predict(x_test)
                # print(cross_val_score(lasso_model, x_train, y_train))

                # for num in range(startTraining, startIQ+1-startTraining):
                #     print(str(x_train[num][featureVectorSize - 2]) +" " + str(
                #         y_train[num]))
                #
                # for num in range(startTraining, len(x_test)):
                #     print(str(x_test[num][featureVectorSize - 2]) +" " + str(
                #         prediction[num]))


                # perform polynome interpolation
                colors = ['teal', 'yellowgreen', 'gold','silver']
                lw = 2
                plt.plot(x_train[:, featureVectorSize - 2], y_train, color='cornflowerblue', linewidth=lw,
                         label="ground truth")
                x_train_plot=np.empty((endTraining-startTraining));
                i=0
                for x in x_train:
                    x_train_plot[i]=x[featureVectorSize - 2]
                    i+=1


                x_test_plot=np.empty((200));
                i=0
                for x in x_test:
                    x_test_plot[i]=x[featureVectorSize - 2]
                    i+=1


                x_plot = np.linspace(0, 10, 200)
                X_plot = x_plot[:, np.newaxis]
                #x_train_plot= [int(x[featureVectorSize - 2]) for x in x_train]
                X_train = x_train_plot[:, np.newaxis]
                X_test = x_test_plot[:, np.newaxis]
                # x_test_plot= [int(x[featureVectorSize - 2]) for x in x_test]
                #y_test_plot= [int(y) for y in prediction]
                #print(X_train)
                #print(y_train)

                for count, degree in enumerate(interpolationDegrees):
                    model = make_pipeline(PolynomialFeatures(degree), scipy.Fitter())
                    model.fit(X_train, y_train)
                    y_plot = model.predict(X_test)
                    fig = plt.figure()
                    #deg2=degree+2
                    plt.plot(x_test_plot, y_plot, color=colors[count], linewidth=lw,
                             label="degree %d" % degree)

                # save synthetic logs into disk

                endMedium=100
                for count,vect in enumerate(x_test[0:endMedium]):
                    for num in vect:
                        output_medium_file.write("%d" % num)
                        output_medium_file.write(" ")
                    output_medium_file.write("%d" % y_plot[count])
                    output_medium_file.write("\n")
                output_medium_file.close()

                # save synthetic logs into disk
                for count, vect in enumerate(x_test[endMedium:]):
                    for num in vect:
                        output_high_file.write("%d" % num)
                        output_high_file.write(" ")
                    output_high_file.write("%d" % y_plot[endMedium+count])
                    output_high_file.write("\n")
                output_high_file.close()
                plt.scatter(x_train[:, featureVectorSize - 2], y_train, color='navy', s=30, marker='o', label="training points")
                plt.legend()
                plt.title(file)
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                #plt.show()
                fig.savefig(os.path.join(outputFigureDirectoryLocation,file +'_plot.png'))
                exit();

if __name__ == "__main__":
   main()