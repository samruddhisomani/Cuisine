# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:58:10 2015

@author: Samruddhi Somani
"""
execfile('Original.py')
execfile('tfidf.py')

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#fitting/examining tree
s2=DecisionTreeClassifier(max_depth=3, random_state=5)
s2.fit(x,cuisine)
leaves2=pd.Series(s2.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves2],axis=1)
m=list(leaves2.value_counts().index.values) #[3, 6, 10, 4, 7, 13, 11, 14]
for y in m:
    print y
    print idk[leaves2==y]['cuisine'].value_counts()
leaves2.value_counts()

#==============================================================================
# 3     33684: SGD SVM
# 6      2991: SGD SVM
# 10     1848: Naive Bayes/Logistic Regression
# 4       914: Naive Bayes/Logistic Regression
# 7       300: Logistic Regression
# 13       20: Naive Bayes
# 11       14: Naive Bayes
# 14        3: Naive Bayes
#==============================================================================

#savings models in a dictionary
m={}
m[3]=SGDClassifier (loss='squared_hinge',random_state=0,n_jobs=2,warm_start=True,n_iter=25)
m[6]=SGDClassifier (loss='squared_hinge',random_state=0,n_jobs=2,warm_start=True,n_iter=25)
m[10]=SGDClassifier (loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=50)
m[4]=SGDClassifier (loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=50)
m[7]=SGDClassifier (loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=50)
m[13]=MultinomialNB()
m[11]=MultinomialNB()
m[14]=MultinomialNB()

#saving leaf data in a dictionary
clean_cuisines={i:idk['cuisine'][idk['leaves']==i] for i in m}
clean_x={i:x[np.array([idk['leaves']==i]).nonzero()[0]] for i in m}

for i in m:
    m[i].fit(clean_x[i],clean_cuisines[i])
    print '\n',i
    print m[i]
    print m[i].score(clean_x[i],clean_cuisines[i])
#3 and 6 are terrible. Everything else is okay.    


#Combine stuff-->More data. This doesn't work.
def ModelMap(leaf):
    
    #these are the hard ones
    if leaf==3 or leaf==6:
        return 1
    #these are the logistic regressions    
    elif leaf==4 or leaf==7 or leaf==10:
        return 2
    #these are the multinomial naive bayes
    elif leaf==13 or leaf==11 or leaf==14:
        return 3
        
M={}
M[1]=RandomForestClassifier(max_depth=1000,n_estimators=2500,warm_start=True,oob_score=True,random_state=0,criterion='gini',n_jobs=2)
M[2]=SGDClassifier (loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=50)
M[3]=MultinomialNB()

idk['map']=idk['leaves'].apply(ModelMap)

CleanC={i:idk['cuisine'][idk['map']==i] for i in [3,2,1]}
CleanX={i:x[np.array([idk['map']==i]).nonzero()[0]] for i in [3,2,1]}


for i in [3,2,1]:
    M[i].fit(CleanX[i],CleanC[i])
    print '\n',i
    print M[i]
    print M[i].score(CleanX[i],CleanC[i])

for i in [1,2,3]:
    print '\n',i
    print M[i]
    print M[i].score(clean_x[i],clean_cuisines[i])

CleanC[2]