# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 10:13:19 2015

@author: Samruddhi Somani
"""

execfile('Original.py')

log=pd.read_csv('../Submissions/sam_log_sub1.csv').set_index('id')
xgb=pd.read_csv('../Submissions/xgb_sub10.csv').set_index('id')
svm=pd.read_csv('../Submissions/svr_sub1.csv').set_index('id')
nb=pd.read_csv('../Submissions/NB_predictions.csv').set_index('id')
pca=pd.read_csv('../Submissions/sam_pca_sub1.csv').set_index('id')

df=pd.concat([log,xgb,svm,nb,pca],axis=1)

df.columns=['log','xgb','svm','nb','pca']

#computing unaminity
mask=df.apply(lambda x: min(x) == max(x),1)
mask.mean()

#calculating mode. some have two.
predictions=df.mode(axis=1)

priors=cuisine.value_counts()

def winner(x):
    
    if pd.isnull(x[1]):
        return x[0]
    
    else:
              
        p1=priors.loc[x[0]]
        p2=priors.loc[x[1]]

        if p1>p2:
            return x[0]
        elif p1<p2:
            return x[1]
   
answers=predictions.apply(winner,axis=1)
