'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


import numpy


batch_size = 128
num_classes = 3834092
epochs = 5

featureVectorSize = 213
traningDataNumber = 1000

# input image dimensions
#img_rows, img_cols = 28, 28
img_rows, img_cols = 213, 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = numpy.array([[[0 for x in range(featureVectorSize)] for y in range(10)]for y in range(traningDataNumber)],numpy.float32)
y_train = numpy.array([0 for x in range(traningDataNumber)],numpy.float32)


with open(r'logExecutionPlan') as f:
    a = f.read().splitlines()
    for i in range(traningDataNumber):
        imgI = numpy.loadtxt(a[i * 10 + i:((i + 1) * 10) + i], comments="#", delimiter=" ", unpack=False)
        # imgI = inputFile[i * 10 + i:((i + 1) * 10) + i - 1]
        x_train[i] = imgI
        y_train[i] = a[((i + 1) * 10) + i]
        # inputFile[((i + 1) * 10) + i, 0:featureVectorSize]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Conv2D(320, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(1, kernel_initializer='normal'))


model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

estimators = []
#estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=1,validation_data=(x_train, y_train))))

pipeline = Pipeline(estimators)
#pipeline.fit(x_train,y_train)
#score = pipeline.score(x_train,y_train)

model.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_data=(x_train, y_train))
score = model.evaluate(x_train,y_train)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


prediction = model.predict_classes(x_train)
for num in range(1, 34):
    if num % 2 == 0:
        print("estimated time for java "+ str(prediction[num]))
    else:
        print("estimated time for spark "+ str(prediction[num]))



for num in range(1, 34):
    if num % 2 == 0:
        print("estimated time for " + str(x_train[num][featureVectorSize - 2]) + "-" + str(
            x_train[num][featureVectorSize - 1]) + " in java : " + str(prediction[num]) + "(real " + str(
            y_train[num]) + ")")
    else:
        print("estimated time for " + str(x_train[num][featureVectorSize - 2]) + "-" + str(
            x_train[num][featureVectorSize - 1]) + " in spark : " + str(prediction[num]) + "(real " + str(
            y_train[num]) + ")")

