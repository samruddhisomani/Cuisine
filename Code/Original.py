from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train=pd.read_json("../Data/train.json")
w=pd.read_csv("../Data/weights4.csv",index_col=0)
weights=w['weight'].to_dict()

ingredients=train['ingredients']
cuisine=train['cuisine']

def no_tokenizer(doc):
    return doc

v=CountVectorizer(preprocessor=None,tokenizer=None,analyzer=no_tokenizer)

x=v.fit_transform(ingredients)

test=pd.read_json("../Data/test.json")

ingredientst=test['ingredients']

xtest=v.transform(ingredientst)

