# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:44:10 2016

@author: shrestha

THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32 python Model.py
"""

### This model has been inspired by the model suggested by Andrej Karpathy for training 
## CIFAR given in the below link
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

## Model with 4 Conv Layer and 2 FC layers

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization 
from keras.regularizers import l2
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import cPickle as pickle
import numpy as np
import h5py

batch_size = 32
nb_classes = 7
nb_epoch = 50
data_augmentation = True



# input image dimensions
img_rows, img_cols = 48, 48

img_channels = 1

XTrain=pickle.load(open('../Data2/XTrain','rb'))

yTrain=pickle.load(open('../Data2/yTrain','rb'))
XTest=pickle.load(open('../Data2/XValid','rb'))
yTest=pickle.load(open('../Data2/yValid','rb'))

XRealTest=pickle.load(open('../Data2/XTest','rb'))
yRealTest=pickle.load(open('../Data2/yTest','rb'))

X_train=np.reshape(XTrain,(len(XTrain),1,48,48))
y_train=np.array(yTrain)
X_test=np.reshape(XTest,(len(XTest),1,48,48))
y_test=np.array(yTest)

#X_train=X_train[:10000]
#y_train=y_train[:10000]
X_RealTest=np.reshape(XRealTest,(len(XRealTest),1,48,48))

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(1024, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.load_weights('Result3/model4.h5')
# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    ModelCheckpoint('Result/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    ModelCheckpoint('Result/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
                        

json_string = model.to_json()
open('Result/model4.json', 'w').write(json_string)
model.save_weights('Result/model4.h5',overwrite=True)

yVpred=model.predict_classes(X_test)
pickle.dump(yVpred,open('Result/yVPredModel4','wb'))

yTestpred=model.predict_classes(X_RealTest)
pickle.dump(yTestpred,open('Result/yTestPredModel4','wb'))

A=[1 for i in range(len(yTestpred)) if yTestpred[i]==yRealTest[i]]
acc=(sum(A)*1.0)/len(X_RealTest)

print ('Acc',acc)

#pickle.dump(hist.history,open('HistoryModel4','wb'))

