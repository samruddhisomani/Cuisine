# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:50:37 2015

@author: Samruddhi Somani
"""

from sklearn.decomposition import TruncatedSVD  
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from numpy import logspace

execfile('tfidf.py')

#training pca

#100: 28%
#500: 62%
#1000: 80%
#1650: 90%

t=TruncatedSVD(n_components=100,random_state=0)

t.fit(x)

t.explained_variance_ratio_.sum()

#Pipeline (create new pipelines to cv over loss/weights)

#Logistic Regression, balanced weights
t=TruncatedSVD(random_state=0)#cv over n_components
s=StandardScaler()
sgd1=SGDClassifier(class_weight='balanced',loss='log',random_state=0,warm_start=True)#cv over n_iters/alpha
balanced_log=Pipeline([('tsvd',t),('scaler',s),('sgd',sgd)])

c=[100,500,1000,1650]
n=[25,50,75]
alpha=logspace(-2,2,5)

params={'tsvd__n_components':c,'sgd__n_iter':n,'sgd__alpha':alpha}

cv=GridSearchCV(balanced_log,params)

cv.fit(x,cuisine)

