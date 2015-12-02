#from _future_ import unicode_literals
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from numpy import logspace
from scipy import shape
from sklearn.externals import joblib
import gc
from sam import SubFile
from seaborn import heatmap
import matplotlib.pyplot as plt
#initializing DTM
execfile('Original.py')

#test/train split
X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.50, random_state=42,stratify=cuisine)

L=LabelEncoder()
y_tr=L.fit_transform(y_train)
y_te=L.transform(y_test)

eta=list(logspace(-2,1,10))


params={'max_depth':[1,2,4],'n_estimators':[10,50,100,250,500,750,1000],'learning_rate':eta}

#depth of 1:71%
qyz=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=10000,
                      max_depth=1,
                      seed=0)
                      
qyz.fit(X_train,y_train,eval_set=[(X_test,y_te)],eval_metric='merror',early_stopping_rounds=100)
answers=qyz.predict(xtest)

joblib.dump(qyz,'qyz.pkl')


#depth of 2: 73%
zyx=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=2500,
                      max_depth=2,
                      seed=0,
                      subsample=.5,
                      colsample_bytree=.5)

zyx.fit(X_train,y_train,eval_set=[(X_test,y_te)],eval_metric='merror',early_stopping_rounds=100)

joblib.dump(zyx,'zyx.pkl')

del zyx

#depth of 3: 75%
lmnop=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=5000,
                      max_depth=3,
                      seed=0,
                      subsample=.5,
                      colsample_bytree=.5)

lmnop.fit(X_train,y_train,eval_set=[(X_test,y_te)],eval_metric='merror',early_stopping_rounds=100)

joblib.dump(lmnop,'lmnop.pkl')

del lmnop

#depth of 3, 7500 trees-->75%: 5000 is best
abc=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=7500,
                      max_depth=3,
                      seed=0,
                      subsample=.5,
                      colsample_bytree=.5)

abc.fit(X_train,y_train,eval_set=[(X_test,y_te)],eval_metric='merror',early_stopping_rounds=100)

joblib.dump(abc,'abc.pkl')

abc=joblib.load('abc.pkl')

answers=abc.predict(x)

#################

qwe=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=5000,
                      max_depth=3,
                      seed=0,
                      subsample=.5,
                      colsample_bytree=.5)
                      
qwe.fit(x,cuisine)
answers=qwe.predict(x)


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
 
 
    h=heatmap(df_new,**kwargs).get_figure()
    ax=h.add_subplot(111)
    ax.set_xlabel('Predictions')
    
    
    
q=ConfusionMatrix(cuisine,pd.Series(answers),'precision')
h=heatmap(q).get_figure()
ax=h.add_subplot(111)
ax.set_xlabel('Predictions')
h.tight_layout()
h.set_size_inches(8,5.5)
h.savefig('../Images/xgb.png',bbox_inches='tight',dpi=100)



y_t=pd.Series(cuisine,name='Truth')
y_p=pd.Series(answers,name='Predictions')
df=pd.crosstab(index=y_t, columns=y_p,dropna=False).fillna(0)

q=df.sum(axis=0)
    
SubFile(test['id'],pd.Series(answers),'../Submissions/xgb_sub10.csv')


#depth of 4, 10000 trees: 75%-->4500 is best

ghi=xgb.XGBClassifier(objective='multi:softmax',
                      nthread=3,
                      learning_rate=.01, 
                      n_estimators=10000,
                      max_depth=4,
                      seed=0,
                      subsample=.5,
                      colsample_bytree=.5)

ghi.fit(X_train,y_train,eval_set=[(X_test,y_te)],eval_metric='merror',early_stopping_rounds=100)

joblib.dump(ghi,'ghi.pkl')

answers=zyx.predict(xtest)

a_s=pd.Series(answers)
sub=pd.concat([test['id'],a_s],axis=1).set_index('id').rename(columns={0:'cuisine'})

sub.to_csv('sam_xgb_sub_4.csv')

#57% with 100 trees (sub1) on test, col/obs subsamplling
#73% with 1000 trees(sub2) on test, cols/obs subsampling
#68% with 5000 trees (sub3) on test, no subsampling



