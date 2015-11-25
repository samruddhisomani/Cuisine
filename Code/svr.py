# -*- coding: utf-8 -*-

execfile('tfidf.py')

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
import sam
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.50, random_state=42)
s=SGDClassifier(n_iter=20,random_state=0)
s.fit(X_train,y_train)
answer=s.predict(X_test)
accuracy_score(y_test,answer)
