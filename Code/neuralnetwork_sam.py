# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 11:22:40 2015

@author: Samruddhi Somani
"""

from sknn.mlp import Classifier,Layer
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
execfile('Original.py')

L=LabelEncoder()
y_=L.fit_transform(cuisine)

nn=Classifier(random_state=0,
              layers=[Layer('Maxout',units=100,pieces=2),
                      Layer('Softmax')],
              batch_size=100)

params={'learning_rate': [.01,.1]
        'n_iter':[10,25,50]}
        
cv=GridSearchCV(nn,params,cv=2,n_jobs=2)

cv.fit(x,cuisine)