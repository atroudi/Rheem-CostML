from __future__ import print_function
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
import sys


batch_size = 128
num_classes = 10
epochs = 1
#featureVectorSize = 105;
featureVectorSize = 213
#nn =
# input image dimensions
img_rows, img_cols = 28, 28

traningDataNumber = 160;

def wider_deep_model():
	# create model
	model = Sequential()
	model.add(Dense(featureVectorSize+20, input_dim=featureVectorSize, kernel_initializer='normal', activation='relu'))
	model.add(Dense(55, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# define mnist model
def mnist_model():
    input_shape = (img_rows, img_cols,1)
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
    return model


def main():
    #sys.argv[2]
    batch_sizes = [64]
    epochs = [500]
    nns=["widedeep"]

    x_train = [[0 for x in range(featureVectorSize)] for y in range(traningDataNumber)]
    y_train = [0 for x in range(traningDataNumber)]
    #x_train = [100][100]
    #nn = sys.argv[3]
    for batch_size in batch_sizes:
        for epoch in epochs:
            for nn in nns:


                # input image dimensions
                img_rows, img_cols = 28, 28

                # the data, shuffled and split between train and test sets
                #(x_train, y_train), (x_test, y_test) = mnist.load_data()

                # the data, shuffled and split between train and test sets
                #(x_train, y_train), (x_test, y_test) = mnist.load_data()
                #inputFile = numpy.genfromtxt("logExecutionPlan", comments="#", delimiter=" ", unpack=False)

                with open(r'logExecutionPlan') as f:
                    a = f.read().splitlines()
                    for i in range(traningDataNumber):
                        imgI = numpy.loadtxt(a[i * 10 + i:((i + 1) * 10) + i - 1], comments="#", delimiter=" ", unpack=False)
                        # imgI = inputFile[i * 10 + i:((i + 1) * 10) + i - 1]
                        x_train[i] = imgI
                        y_train[i] = a[((i + 1) * 10) + i]
                        #inputFile[((i + 1) * 10) + i, 0:featureVectorSize]

                print(x_train)
                print(y_train)

                model = mnist_model()

                estimators = []
                #estimators.append(('standardize', StandardScaler()))
                estimators.append(('mlp', KerasRegressor(build_fn=mnist_model(), epochs=epoch, batch_size=batch_size, verbose=0)))

                pipeline = Pipeline(estimators)

                (x_train, y_train), (x_test, y_test) = mnist.load_data()

                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')
                x_train /= 255
                x_test /= 255

                # convert class vectors to binary class matrices
                y_train = keras.utils.to_categorical(y_train, num_classes)
                y_test = keras.utils.to_categorical(y_test, num_classes)

                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_train, y_train))


                #pipeline.fit(x_train, y_train)
                # score = pipeline.score(x_train, y_train)

                score = model.evaluate(x_train, y_train,batch_size=213, verbose=0)

                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                # create images shapes (rows:[0:10]*N; labels:11*N )
                # for i in range(20):
                #     imgI = inputFile[i*10+i:((i+1)*10)+i-1, 0:featureVectorSize]
                #     #imgI = inputFile[i * 10 + i:((i + 1) * 10) + i - 1]
                #     x_train[i] = imgI
                #     y_train = inputFile[((i+1)*10)+i, 0:featureVectorSize]




if __name__ == "__main__":
   main()