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
from sklearn.linear_model import LogisticRegression
from sam import SubFile

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

n=[25,50,75]
#alpha=logspace(-2,2,5)

###Try again.

n=[25,50,75]
sgdlog=SGDClassifier(loss='log',random_state=0,warm_start=True)
sgdsvm=SGDClassifier(loss='squared_hinge',random_state=0,warm_start=True)
params={'n_iter':n}
s=StandardScaler()


#100 components
t100=TruncatedSVD(random_state=0,n_components=100)
trx=t100.fit_transform(x)
trx=s.fit_transform(trx)

w=LogisticRegression(solver='sag')
w.fit(trx,cuisine)
w.score(trx,cuisine)

cv100_svm=GridSearchCV(sgdsvm,params,cv=2)
cv100_svm.fit(trx,cuisine)
cv100_svm.best_score_

#200 components
t200=TruncatedSVD(random_state=0,n_components=200)
trx=t200.fit_transform(x)
trx=s.fit_transform(trx)

w2=LogisticRegression(solver='sag')
w2.fit(trx,cuisine)
w2.score(trx,cuisine)

#250 components
t25=TruncatedSVD(random_state=0,n_components=250)
trx=t25.fit_transform(x)
trx=s.fit_transform(trx)

w25=LogisticRegression(solver='sag',n_jobs=3,max_iter=100)
w25.fit(trx,cuisine)
w25.score(trx,cuisine)

#500 components
t50=TruncatedSVD(random_state=0,n_components=500)
trx=t50.fit_transform(x)
trx=s.fit_transform(trx)

w50=LogisticRegression(solver='sag',n_jobs=3,max_iter=100)
w50.fit(trx,cuisine)
w50.score(trx,cuisine)

trT=t50.transform(xtest)
trT=s.transform(trT)
a=w50.predict(trT)
a=pd.Series(a)
SubFile(test['id'],a,'../Submissions/sam_pca_sub2.csv') #500 components


#1000 components
t1000=TruncatedSVD(random_state=0,n_components=1000)
trx=t1000.fit_transform(x)
trx=s.fit_transform(trx)

w10=LogisticRegression(solver='sag',n_jobs=2,max_iter=150)
w10.fit(trx,cuisine)
w10.score(trx,cuisine)

trT=t1000.transform(xtest)
trT=s.transform(trT)
a=w10.predict(trx)
a=pd.Series(a)

SubFile(test['id'],a,'../Submissions/sam_pca_sub1.csv') #1000 components


cv1000_svm=GridSearchCV(sgdsvm,params,cv=2)
cv1000_svm.fit(trx,cuisine)
cv1000_svm.best_score_

t1650=TruncatedSVD(random_state=0,n_components=1650)
trx=t1650.fit_transform(x)
trx=s.fit_transform(trx)

cv1650_svm=GridSearchCV(sgdsvm,params,cv=2)
cv1650_svm.fit(trx,cuisine)
cv1650_svm.best_score_


#confusion matrix
import sam

q=ConfusionMatrix(cuisine,a,'precision')
hmwrapper(q,'../Images/pca.png')

def ConfusionMatrix(y_true,y_pred,kind):

    y_t=pd.Series(y_true,name='Truth')
    y_p=pd.Series(y_pred,name='Predictions')
    df=pd.crosstab(index=y_t, columns=y_p,dropna=False).fillna(0)

    if kind=='original':
        df_new=df
    elif kind=='precision':
        #how many of selected items are relevant:
        #divide row over sum of columns
        df_new=df.div(df.sum(axis=1),axis='index').fillna(0)
    elif kind=='recall':
        #how many relevant items are selected:
        #divide each column over sum of rows
        df_new=df.div(df.sum(axis=0),axis='columns').fillna(0)
    return df_new
    
def hmwrapper(cm,filename):
    h=heatmap(cm).get_figure()
    ax=h.add_subplot(111)
    ax.set_xlabel('Predictions')
    h.tight_layout()
    h.set_size_inches(8,5.5)
    h.savefig(filename,bbox_inches='tight',dpi=100)
