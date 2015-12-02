# -*- coding: utf-8 -*-

from scipy.io import mmwrite,mmread
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sam import ConfusionMatrix,SubFile
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from numpy import logspace
from sklearn.grid_search import GridSearchCV
from seaborn import heatmap


execfile('Original.py')

#trying sgd: 76% accuracy, 78% on test
X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.50, random_state=42,stratify=cuisine)

params={'n_iter':[5,10,20,50,100,150,200,300,500]}
s=SGDClassifier(random_state=0,warm_start=True)
cvSGD=GridSearchCV(s,params,cv=2)
cvSGD.fit(X_train,y_train)
cvSGD.best_estimator_
answer=cvSGD.predict(X_test)
accuracy_score(y_test,answer)
heatmap(ConfusionMatrix(y_test,answer,'recall'))
 
#fitting on full training data 
params={'n_iter':[5,10,20,50,100,150,200,300,500]}
sFull=SGDClassifier(random_state=0,warm_start=True)  
cvSGD=GridSearchCV(sFull,params,cv=2)
cvSGD.fit(x,cuisine)
print cvSGD.best_estimator_
print cvSGD.best_score_
answers=cvSGD.predict(xtest)
SubFile(test['id'],pd.Series(answers),'svr_sub1.csv')
print cvSGD.grid_scores_

#honing in on cross val: This didn't work. Use 50.

params={'n_iter': [45,46,47,48,49,50,51,52,53,54,55]}
cvSGD1=GridSearchCV(sFull,params,cv=2)
cvSGD1.fit(x,cuisine)
print cvSGD1.best_estimator_
print cvSGD1.best_score_
cvSGD1.grid_scores_
answers=cvSGD1.predict(xtest)
SubFile(test['id'],pd.Series(answers),'svr_sub2.csv')
print cvSGD.grid_scores_

#trying svc

c=logspace(-3,3,num=10)
params={'C':c}

#fitting with square hinge loss, unbalanced classes: best accuracy is 75% on training
w=LinearSVC(random_state=0)

cv=GridSearchCV(w,params,n_jobs=1,cv=2)

cv.fit(X_train,y_train)

cv.best_score_

#fitting with hinge loss, unbalanced classes: best accuracy is 75% on training
wM=LinearSVC(random_state=0,loss='hinge',n_iter=50)

cvM=GridSearchCV(wM,params,n_jobs=1)

cvM.fit(X_train,y_train)

cvM.best_score_

#final fitting
sFull=SGDClassifier(random_state=0,warm_start=True,n_iter=50)
sFull.fit(x,cuisine)

answers=sFull.predict(x)


#confusion matrix building

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


q=ConfusionMatrix(cuisine,pd.Series(answers),'precision')
hmwrapper(q,'../Images/sam_svm.png')
