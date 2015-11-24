import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from __future__ import unicode_literals
from sam import balanced_samples
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train=pd.read_json("../Data/train.json")

ingredients=train['ingredients']
cuisine=train['cuisine']

def no_tokenizer(doc):
    return doc

v=CountVectorizer(preprocessor=None,tokenizer=None,analyzer=no_tokenizer,min_df=5)

x=v.fit_transform(recipes)

v.get_feature_names()

dfO=pd.concat([cuisine,ingredients],axis=1)

cuisines=list(cuisine.unique())

df=balanced_samples(2000,cuisines,dfO,['cuisine','ingredients'],'cuisine')

test=dfO[~dfO.index.isin(df.index.values)]

q=v.fit_transform(df['ingredients'])

w=LogisticRegression()
w.fit_transform(q,y=df['cuisine'])
t=v.transform(test['ingredients'])
p=w.predict(t)

accuracy_score(test['cuisine'],p)

#500: 62%
#1000: 72%
#1500: 73%
#2000: 74%
