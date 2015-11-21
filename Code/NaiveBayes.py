from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#import data
train=pd.read_json("../Data/train.json")
test=pd.read_json("../Data/test.json")

recipes=train['ingredients']
cuisine=train['cuisine']


def no_tokenizer(doc):
    return doc

v=CountVectorizer(preprocessor=None,tokenizer=None,analyzer=no_tokenizer) #define countvectorizer with changed analyzer

x=v.fit_transform(recipes) #make document term matrix

v.get_feature_names()


#multinomial Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.33, random_state=42) #split into test and train

clf = MultinomialNB()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)

accuracy_score(y_test,pred)

#function to make list of ingredients and turn it into a string with tokens delimited by comma
def stringer(x):
    s=','
    return s.join(x)

train['ing_string']=train['ingredients'].apply(stringer)
test['ing_string']=test['ingredients'].apply(stringer)
recipe2=train['ing_string']

recipe2

#Naive Bayes with Document Term Matrix where each word is token
cv2=CountVectorizer()
dtm=cv2.fit_transform(recipes2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(dtm, cuisine, test_size=0.33, random_state=42)
clf2=MultinomialNB()
clf2.fit(X_train2, y_train2)
pred2=clf2.predict(X_test2)
accuracy_score(y_test2,pred2)

cv2.get_feature_names()