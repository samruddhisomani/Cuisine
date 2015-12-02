from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import textblob as tb

train=pd.read_json("../Data/train.json")

ingredients=train['ingredients']
cuisine=train['cuisine']

def Tokenizer(doc):
    #lower case alphabetized list of each ingredient
    s=[sorted(list(tb.TextBlob(q.lower()).words.lemmatize())) for q in doc]
    z=[' '.join(x) for x in s]
    return z

v=TfidfVectorizer(preprocessor=None,tokenizer=None,analyzer=Tokenizer,min_df=5)

x=v.fit_transform(ingredients)

#v.get_feature_names()