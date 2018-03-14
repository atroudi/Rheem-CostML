'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import numpy
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from numpy import loadtxt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import pickle

batch_size = 128
num_classes = 10
epochs = 1
#featureVectorSize = 105;
#featureVectorSize = 146;
featureVectorSize = 251;
#nn =
# input image dimensions
img_rows, img_cols = 1, 105

# define mnist model
def mnist_model():
    input_shape = (1, featureVectorSize, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(featureVectorSize, input_dim=featureVectorSize, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(featureVectorSize, input_dim=featureVectorSize, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(featureVectorSize+20, input_dim=featureVectorSize, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def wider_deep_model_old():
	# create model
	model = Sequential()
	model.add(Dense(featureVectorSize+20, input_dim=featureVectorSize, kernel_initializer='normal', activation='relu'))

	model.add(Dense(featureVectorSize//2, kernel_initializer='normal', activation='relu'))

	model.add(Dense(55, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def wider_deep_model():
    # create model
    model = Sequential()
    # model.add(
    #     Dense(featureVectorSize + 20, input_dim=featureVectorSize, activation='relu'))
    # model.add(Dropout(0.2))

    # layer 1
    model.add(Dense(featureVectorSize + 20,input_dim=featureVectorSize, kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # layer 2
    model.add(Dense(featureVectorSize//2, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # layer 3
    model.add(Dense(featureVectorSize // 4, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # layer 3
    model.add(Dense(1, kernel_initializer='normal'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main():
    #sys.argv[2]
    batch_sizes = [128]
    # epochs = [500,1000,1500]
    epochs = [500]
    nns=["widedeep"]
    #nn = sys.argv[3]
    for batch_size in batch_sizes:
        for epoch in epochs:
            for nn in nns:
                # the data, shuffled and split between train and test sets
                #(x_train, y_train), (x_test, y_test) = mnist.load_data()
                #inputFile = loadtxt("resources/planVector_newShape.txt", comments="#", delimiter=" ", unpack=False)
                #inputFile = loadtxt("resources/planVector_1D_java_spark_javaSpark_withplatformError.log", comments="#", delimiter=" ", unpack=False)
                inputFile = loadtxt("resources/planVector_1D_231_221_merge_7_withSynthetic.log", comments="#", delimiter=" ", unpack=False)


                start = 64
                x_train = inputFile[start:, 0:featureVectorSize]
                y_train = inputFile[start:, featureVectorSize]

                x_test = inputFile[:start, 0:featureVectorSize]
                y_test = inputFile[:start, featureVectorSize]

                print(x_train)
                print(y_train)

                # if K.image_data_format() == 'channels_first':
                #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
                #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
                #     input_shape = (1, img_rows, img_cols)
                # else:
                #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
                #     input_shape = (img_rows, img_cols, 1)

                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')
                # x_train /= 255
                # x_test /= 255
                print('x_train shape:', x_train.shape)
                print(x_train.shape[0], 'train samples')
                print(x_test.shape[0], 'test samples')

                # convert class vectors to binary class matrices
                # y_train = keras.utils.to_categorical(y_train, num_classes)
                # y_test = keras.utils.to_categorical(y_test, num_classes)

                #model = mnist_model()

                # fix random seed for reproducibility
                seed = 7
                numpy.random.seed(seed)
                # evaluate model with standardized dataset
                #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

                estimators = []
                estimators.append(('standardize', StandardScaler()))
                if nn=="wide":
                    estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=epoch, batch_size=batch_size, verbose=0)))
                elif nn=="deep":
                    estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=epoch, batch_size=batch_size, verbose=0)))
                elif nn=="widedeep":
                    model = KerasRegressor(build_fn=wider_deep_model, epochs=epoch, batch_size=batch_size, verbose=0)
                    model2 =RandomForestRegressor(max_depth=50, random_state=0)
                    estimators.append(('mlp', model))

                print("Results with nn:"+nn+" and epoch:" + str(epoch) + "-batchsize:" + str(batch_size))

                #kfold = KFold(n_splits=10, random_state=seed)
                #results = cross_val_score(estimator, x_train, y_train, cv=kfold)

                pipeline = Pipeline(estimators)
                pipeline.fit(x_train,y_train)

                score = pipeline.score(x_train,y_train)
                print("Results: %.2f (%.2f) MSE" % (score, score))


                # save the pipeline model to disk
                filename = 'nnModel'+'_batchSize'+str(batch_size)+"_epoch"+str(epoch)+".pkl"
                #joblib.dump(model2, filename)

                # Save the Keras model first:
                pipeline.named_steps['mlp'].model.save('keras_model'+'_batchSize'+str(batch_size)+"_epoch"+str(epoch)+'.h5')

                # This hack allows us to save the sklearn pipeline:
                pipeline.named_steps['mlp'].model = None

                # Finally, save the pipeline:
                joblib.dump(pipeline, filename)

                # Load the pipeline first:
                pipeline = joblib.load(filename)

                # Then, load the Keras model:
                pipeline.named_steps['mlp'].model = load_model('keras_model'+'_batchSize'+str(batch_size)+"_epoch"+str(epoch)+'.h5')

                # pickle.dump(pipeline, open(filename, 'wb'))

                # serialize model to JSON
                # model_json = pipeline.to_json()
                # with open(filename+".json", "w") as json_file:
                #     json_file.write(model_json)
                # # serialize weights to HDF5
                # pipeline.save_weights(filename+".h5")
                # print("Saved model to disk")

                #for num in [1,12]:
                #    print(estimator.predict(x_train[num]))
                #print(estimator.predict(x_train[1]))
                
                prediction = pipeline.predict(x_test)
                for num in range(0,start):
                    if num%2==0:
                        print ("estimated time for "+ str(x_test[num][featureVectorSize-2])+"-"+str(x_test[num][featureVectorSize-1]) + " in java : %.2f " % (prediction[num]) +"(real "+ str(y_test[num])+ ")" )
                    else:
                        print ("estimated time for "+ str(x_test[num][featureVectorSize-2])+"-"+str(x_test[num][featureVectorSize-1]) + " in spark : %.2f " % (prediction[num]) +"(real "+ str(y_test[num])+ ")" )

                #kfold = KFold(n_splits=10, random_state=seed)
                #results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
                #accuracy_score(y_test, prediction)
                #print("Results: %.2f (%.2f) MSE" % (estimators., prediction.std()))

                #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

                # model.fit(x_train, y_train,
                #           batch_size=batch_size,
                #           epochs=epochs,
                #           verbose=1,
                #           validation_data=(x_test, y_test))

                #score = model.evaluate(x_test, y_test, verbose=0)
                #print('Test loss:', score[0])
                #print('Test accuracy:', score[1])

if __name__ == "__main__":
   main()
