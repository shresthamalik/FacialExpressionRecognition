# -*- coding: utf-8 -*-
"""
Created on Thu May 26 21:11:42 2016

@author: shrestha
"""

### PreProcessing
import csv
import numpy as np
import cPickle as pickle
#import matplotlib.pyplot as plt
#import random as rand
import math

'''
XTrain=pickle.load(open('XTrain','rb'))
yTrain=pickle.load(open('yTrain','rb'))
XValid=pickle.load(open('XValid','rb'))
yValid=pickle.load(open('yValid','rb'))
XTest=pickle.load(open('XTest','rb'))
yTest=pickle.load(open('yTest','rb'))
'''

##Subtracting mean from image
def removeDC(X):
    mean=np.mean(X,axis=1)
    mean=np.mean(mean).transpose()
    X=X-mean
    return X
 

def normaliseAndFit(Xtrain,X):
    mean=np.mean(Xtrain,axis=0)
    std=np.std(Xtrain,axis=0)
    Xtrain=(Xtrain-mean)/std
    X=(X-mean)/std
    return Xtrain,X

#XTrain=np.reshape(XTrain,(48,48))

mean=np.mean(XTrain)

'''
XTrain=removeDC(XTrain)
XValid=removeDC(XValid)

XTrain,XValid=normaliseAndFit(XTrain,XValid)


pickle.dump(XTrain,open('XNormTrain','wb'))
#pickle.dump(yTrain,open('yTrain','wb'))
pickle.dump(XValid,open('XNormValid','wb'))
#pickle.dump(yValid,open('yValid','wb'))
#pickle.dump(XTest,open('XTest','wb'))
#pickle.dump(yTest,open('yTest','wb'))
'''