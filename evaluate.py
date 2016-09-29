# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:21:48 2016

@author: shrestha
"""

import cPickle as pickle
import numpy as np
## Reading data
H=pickle.load(open('HistoryModel2','rb'))

chk=pickle.load(open('../ShuffleData/CountTrain','rb'))
L=list(chk)
chk=chk/sum(L)
#ypred=pickle.load(open('yPredModel2','rb'))




'''
pickle.dump(ypred,open('yPredModel1','wb'))
pickle.dump(hist.history,open('History','wb'))
'''