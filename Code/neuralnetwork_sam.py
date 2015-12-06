# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 11:22:40 2015

@author: Samruddhi Somani
"""

from sknn.mlp import Classifier,Layer
from sklearn.preprocessing import LabelEncoder
execfile('Original.py')

L=LabelEncoder()
y_=L.fit_transform(cuisine)

nn=Classifier(random_state=0,
              layers=[Layer('Maxout',units=100,pieces=2),
                      Layer('Softmax')],
            n_iter=25,
            batch_size=100)
nn.fit(x,y_)

nn.score(x,y_)