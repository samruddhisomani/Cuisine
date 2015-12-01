# -*- coding: utf-8 -*-

from scipy.io import mmwrite,mmread
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sam import ConfusionMatrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from numpy import logspace
from sklearn.grid_search import GridSearchCV
from seaborn import heatmap


execfile('Original.py')

#trying sgd: 76% accuracy
X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.50, random_state=42,stratify=cuisine)

params={'n_iter':[5,10,20,50]}
s=SGDClassifier(random_state=0,warm_start=True)
cvSGD=GridSearchCV(s,params,cv=2)
cvSGD.fit(X_train,y_train)
answer=cvSGD.predict(X_test)
accuracy_score(y_test,answer)


heatmap(ConfusionMatrix(y_test,answer,'recall'))
  

#trying svc

c=logspace(-3,3,num=100)
params={'C':c}

#fitting with square hinge loss: best accuracy is 75% on training
w=LinearSVC(class_weight='balanced', random_state=0)

cv=GridSearchCV(w,params,n_jobs=-1)

cv.fit(X_train,y_train)

cv.best_score_

#fitting with hinge loss: best accuracy is 74% on training
wM=LinearSVC(class_weight='balanced', random_state=0,loss='hinge')

cvM=GridSearchCV(wM,params,n_jobs=-1)

cvM.fit(X_train,y_train)

cvM.best_score_

