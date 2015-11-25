from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# train=pd.read_json("../Data/train.json")
# test=pd.read_json("../Data/test.json")
#
# recipes=train['ingredients']
# cuisine=train['cuisine']
#
# def no_tokenizer(doc):
#     return doc
#
# v=CountVectorizer(preprocessor=None,tokenizer=None,analyzer=no_tokenizer)
#
# x=v.fit_transform(recipes)
#
# v.get_feature_names()

execfile('Original.py')

X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)

accuracy_score(y_test,pred)
