# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:11:17 2015

@author: Samruddhi Somani
"""

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sam import SubFile

execfile('Original.py')

w=LogisticRegressionCV(cv=3,verbose=1,solver='sag',random_state=0,n_jobs=2)

w.fit(x,cuisine)

right=w.C_[0]

m=LogisticRegression(C=right,solver='liblinear',random_state=5)

m.fit(x,cuisine)
m.score(x,cuisine)

answers=m.predict(x)

SubFile(test['id'],pd.Series(answers),'sam_log_sub2.csv')

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
    h.set_size_inches(8,6)
    h.savefig(filename,bbox_inches='tight',dpi=200)
    
q=ConfusionMatrix(cuisine,pd.Series(answers),'precision')
hmwrapper(q,'../Images/sam_log.png')