from __future__ import unicode_literals
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from seaborn import heatmap
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

def ConfusionMatrix(y_true,y_pred,kind):

    y_t=pd.Series(y_true,name='Truth')
    y_p=pd.Series(y_pred,name='Predictions')
    df=pd.crosstab(index=y_t, columns=y_p).fillna(0)

    if kind=='original':
        df_new=df
    elif kind=='precision':
        #how many of selected items are relevant:
        #divide over sum of columns
        df_new=df.div(df.sum(axis=1)).fillna(0)
    elif kind=='recall':
        #how many relevant items are selected:
        #divide over sum of rows
        df_new=df.div(df.sum(axis=0)).fillna(0)
    
    return df_new
    
def counter(array):
    df=pd.DataFrame(array)
    df.columns=['name']
    return df['name'].value_counts()
    
execfile('Original.py')



#train/test split
X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.33, random_state=42)

#Naive Baye's
clf = MultinomialNB()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)

pred_counts=counter(pred)
y_vc=y_test.value_counts()
y_vc=pd.DataFrame(y_vc)
y_vc.columns=['actual']
y_vc['predicted']=pred_counts
y_vc.plot(kind='bar')


def bargraph(preds,ytest):
    pred_counts=counter(preds)
    y_vc=ytest.value_counts()
    y_vc=pd.DataFrame(y_vc)
    y_vc.columns=['actual']
    y_vc['predicted']=pred_counts
    return y_vc.plot(kind='bar')




NB_CM=confusion_matrix(y_test,pred)
heatmap(NB_CM,xticklabels=labels.index,yticklabels=labels.index)

labels=ConfusionMatrix(y_test,pred,'original')

pd.DataFrame(NB_CM/NB_CM.sum(axis=1).astype(float)).to_csv('percent.csv')
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis(solver='svd')
lda.fit(X_train.toarray(),y_train)

pred_lda=lda.predict(X_test)

lda_cm=confusion_matrix(y_test,pred_lda)
heatmap(lda_cm,xticklabels=labels.index,yticklabels=labels.index)

pred_ldacount=counter(pred_lda)
y_vc=y_test.value_counts()
y_vclda=pd.DataFrame(y_vc)
y_vclda.columns=['actual']
y_vclda['predicted']=pred_ldacount
y_vclda.plot(kind='bar')
   



