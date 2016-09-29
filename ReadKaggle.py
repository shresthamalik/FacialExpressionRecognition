# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:53:31 2016

@author: shrestha
"""


### Reading the Facial Expressions Kaggle Dataset

import csv
import numpy as np
import cPickle as pickle
#import matplotlib.pyplot as plt
#import random as rand
import math


dataheader=[]
XTest=[]
yTest=[]
XTrain=[]
yTrain=[]
XValid=[]
yValid=[]

Emotion=['Angry', 'Disgust', 'Fear', 'Happy','Sad','Surprise','Neutral']
i=0
with open('Data/fer2013.csv', 'rb') as f:
    reader = csv.reader(f)
    ind=0
    for row in reader:
        if ind==0:
            dataheader.append(row)
            ind+=1
        else:
            if(row[2]=='Training'):
                yTrain.append(int(row[0]))
                XTrain.append([int(j) for j in row[1].split()])
            elif(row[2]=='PublicTest'):
                yValid.append(int(row[0]))
                XValid.append([int(j) for j in row[1].split()])
            else:
                yTest.append(int(row[0]))
                XTest.append([int(j) for j in row[1].split()])


XTrain=np.array(XTrain)
XTest=np.array(XTest)
XValid=np.array(XValid)


pickle.dump(XTrain,open('XTrain','wb'))
pickle.dump(yTrain,open('yTrain','wb'))
pickle.dump(XValid,open('XValid','wb'))
pickle.dump(yValid,open('yValid','wb'))
pickle.dump(XTest,open('XTest','wb'))
pickle.dump(yTest,open('yTest','wb'))
pickle.dump(Emotion,open('emotionList','wb'))


