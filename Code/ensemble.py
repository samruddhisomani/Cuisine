# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 10:13:19 2015

@author: Samruddhi Somani
"""

execfile('Original.py')

from sam import SubFile

log=pd.read_csv('../Submissions/sam_log_sub1.csv').set_index('id')
xgb=pd.read_csv('../Submissions/xgb_sub10.csv').set_index('id')
svm=pd.read_csv('../Submissions/svr_sub1.csv').set_index('id')
nb=pd.read_csv('../Submissions/NB_predictions.csv').set_index('id')
pca=pd.read_csv('../Submissions/sam_pca_sub1.csv').set_index('id')
lda=pd.read_csv('../Submissions/lda_preds.csv').set_index('id')
sgd_svm=pd.read_csv('../Submissions/leon_sgd_hinge.csv').set_index('id')

df=pd.concat([log,xgb,svm,nb,pca,lda,sgd_svm],axis=1)

df.columns=['log','xgb','svm','nb','pca','lda','sgd_svm']

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
   
def loser(x):
    
    if pd.isnull(x[1]):
        return x[0]
     
    elif pd.notnull(x[1]) and pd.isnull(x[2]):
        p1=priors.loc[x[0]]
        p2=priors.loc[x[1]]

        if p1<p2:
            return x[0]
        elif p2<p1:
            return x[1]
        
    else:        
        p1=priors.loc[x[0]]
        p2=priors.loc[x[1]]
        p3=priors.loc[x[2]]

        if p1<p2 and p1<p3:
            return x[0]
        elif p2<p1 and p2<p3:
            return x[1]
        elif p3<p1 and p3<p2:
            return x[2]
            
def winner2(x):
    
    if pd.isnull(x[1]):
        return x[0]
     
    elif pd.notnull(x[1]) and pd.isnull(x[2]):
        p1=priors.loc[x[0]]
        p2=priors.loc[x[1]]

        if p1<p2:
            return x[1]
        elif p2<p1:
            return x[0]
        
    else:        
        p1=priors.loc[x[0]]
        p2=priors.loc[x[1]]
        p3=priors.loc[x[2]]

        if p1>p2 and p1>p3:
            return x[0]
        elif p2>p1 and p2>p3:
            return x[1]
        elif p3>p1 and p3>p2:
            return x[2]
            
answers=predictions.apply(winner,axis=1)
<<<<<<< Updated upstream

SubFile(test['id'].reset_index(),pd.Series(answers),'../Submissions/ensemble.csv')

pd.concat([test['id'],answers],axis=1)
=======
answer2=predictions.apply(loser,axis=1)
answer3=predictions.apply(winner2,axis=1)
#answer3.to_csv('../Submissions/ensemble_winner2.csv',index_label='id',header=['cuisine'])
>>>>>>> Stashed changes
