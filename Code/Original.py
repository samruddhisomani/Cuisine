import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from __future__ import unicode_literals

train=pd.read_json("../Data/train.json")

recipes=train['ingredients']
cuisine=train['cuisine']

def no_tokenizer(doc):
    return doc

v=CountVectorizer(preprocessor=None,tokenizer=None,analyzer=no_tokenizer)

x=v.fit_transform(recipes)

v.get_feature_names()
