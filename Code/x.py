import xgboost as xgb

execfile('tfidf.py')

cuisine.values

dtrain=xgb.DMatrix(x,label=cuisine.values)
param={'bst.subsample':0.5,'bst.colsample_bytree':0.5,\
'objective':'multi:softmax','num_classes':20,'nthread':2}

qyz=xgb.XGBClassifier(max_depth=1,n_estimators=1000,objective='multi:softmax')

qyz.fit(x,cuisine)

x

test=pd.read_json("../Data/test.json")

ingredients=test['ingredients']
#cuisine=test['cuisines']

def no_tokenizer(doc):
    return doc

#v=TfidfVectorizer(preprocessor=None,tokenizer=None,analyzer=no_tokenizer,min_df=5)

xtest=v.transform(ingredients)

answers=qyz.predict(xtest)

a_s=pd.Series(answers)
sub=pd.concat([test['id'],a_s],axis=1).set_index('id').rename(columns={0:'cuisine'})

sub.to_csv('sam_xgb_sub_2.csv')

#57% with 100 trees (sub1) on test
#73% with 1000 trees(sub2) on test
qyz.save_model('xbMODEL')
