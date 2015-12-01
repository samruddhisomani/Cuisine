# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:58:10 2015

@author: Samruddhi Somani
"""
execfile('Original.py')
#execfile('tfidf.py')

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sam import SubFile


#fitting/examining tree
s2=DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=500, random_state=5)
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

m_dict3={i:SGDClassifier(loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=150) for i in m}
clean_cuisines={i:idk['cuisine'][idk['leaves']==i] for i in m}
clean_x={i:x[np.array([idk['leaves']==i]).nonzero()[0]] for i in m}

m_dict3[17]=SGDClassifier(loss='hinge',random_state=0,n_jobs=2,warm_start=True,n_iter=50,class_weight='balanced')
m_dict3[8]=SGDClassifier(loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=100)
m_dict3[4]=SGDClassifier(loss='log',random_state=0,n_jobs=2,warm_start=True,n_iter=150)

for i in m:
    print '\n',i
    m_dict3[i].fit(clean_x[i],clean_cuisines[i])
    #print m_dict3[i]
    print m_dict3[i].score(clean_x[i],clean_cuisines[i])
    
df=pd.DataFrame(index=test['id'],columns=m).reset_index()

for i in m:
    df[i]=pd.Series(m_dict3[i].predict(xtest))

df['leaf']=pd.Series(s2.apply(xtest))

df['p']=pd.Series(df.lookup(df.index.values,df['leaf']))

SubFile(df['id'],df['p'],'../Submissions/coop_sub1.csv')



SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=100, n_jobs=2,
       penalty='l2', power_t=0.5, random_state=0, shuffle=True, verbose=0,
       warm_start=True)
       
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=100, n_jobs=2,
       penalty='l2', power_t=0.5, random_state=0, shuffle=True, verbose=0,
       warm_start=True)
    
hard=[14,17,18,4,16]

for i in hard:
    m_dict[i]=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=2000,
                      max_depth=3,
                      seed=0,
                      colsample_bytree=.5)
                      
df=pd.DataFrame(index=test['id'],columns=m)
#this guesses the same thing for every column                                

#take 2
t=TruncatedSVD(random_state=0,n_components=500)#cv over n_components
s=StandardScaler()
clean=Pipeline([('tsvd',t),('scaler',s)])

newX=clean.fit(x)

new=cuisine.value_counts()/cuisine.count()
newr=np.ravel(new)

a=m_dict3[17].predict(x)
pd.Series(a).value_counts()

for i in hard:
    m_dict2[i]=MultinomialNB()

for i in m:
    print '\n',i
    m_dict2[i].fit(clean_x[i],clean_cuisines[i])
    print m_dict2[i]
    print m_dict2[i].score(clean_x[i],clean_cuisines[i])
    df[i]=m_dict2[i].predict(xtest)
    
df=pd.DataFrame(index=test['id'],columns=m)