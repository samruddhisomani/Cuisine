<<<<<<< HEAD
from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import mmwrite

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

execfile('Preprocessed.py')
execfile('Original.py')

X_train, X_test, y_train, y_test = train_test_split(x, cuisine, test_size=0.33, random_state=42,stratify=cuisine)

clf = MultinomialNB()
clf.fit(X_train, y_train)
pred=clf.predict(X_test)

accuracy_score(y_test,pred)

#mmwrite('preprocessed.mtx', x,precision=10)
=======
<<<<<<< HEAD
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

accuracy_score(y_test,pred) #0.72779216821575499

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
dtm=cv2.fit_transform(recipe2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(dtm, cuisine, test_size=0.33, random_state=42)
clf2=MultinomialNB()
clf2.fit(X_train2, y_train2)
pred2=clf2.predict(X_test2)
accuracy_score(y_test2,pred2) #0.72398293463355168

cv2.get_feature_names()

#logistic Regression
logreg = linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
predlog=logreg.predict(X_test)
accuracy_score(y_test,predlog) #0.77815023617248213

#downsample
def dnsmpl(xtrain,ytrain,pos_ratio):
    if pos_ratio >1 or pos_ratio <0:
        print 'the positive ratio must be between 0 and 1'
    else:
        posmask=ytrain.ix[:,0]==1
        negmask=ytrain.ix[:,0]==0
        neg_class=ytrain[negmask]
        pos_class=ytrain[posmask]
        pos_count=float(len(ytrain[posmask]))
        neg_count=float(len(ytrain[negmask]))
        prop_pos=float(pos_count/len(ytrain))
        new_poscount=int(round(pos_ratio*len(ytrain))+1)
        new_samplesize=len(pos_class)/pos_ratio
        neg_samplesize=int(new_samplesize*(1-pos_ratio))
        neg_index=list(neg_class.index.values)
        samplenum=random.sample(set(neg_index), neg_samplesize)
        new_negclass=neg_class.ix[samplenum]
        concat_id=[pos_class,new_negclass]
        downsampled_ytrain=pd.concat(concat_id).sort_index()
        sampled_xtrain=xtrain.ix[downsampled_ytrain.index.values].sort_index()
        dnsmpl_df=pd.merge(downsampled_ytrain,sampled_xtrain,left_index=True,right_index=True)
        
    return dnsmpl_df


#upsample
def upsample(xtrain,ytrain,pos_ratio):
    if pos_ratio >1 or pos_ratio <0:
        print 'the positive ratio must be between 0 and 1'
    else:
        posmask=ytrain.ix[:,0]==1
        negmask=ytrain.ix[:,0]==0
        neg_class=ytrain[negmask]
        pos_class=ytrain[posmask]
        pos_count=float(len(ytrain[posmask]))
        neg_count=float(len(ytrain[negmask]))
        pos_samples_needed=round((pos_ratio*len(ytrain)-len(pos_class))/(1-pos_ratio))
        samples=list(np.random.choice(pos_class.index.values,pos_samples_needed,replace=True))
        new_ytrain=ytrain
        new_xtrain=xtrain
        new_df=pd.merge(new_ytrain,new_xtrain,left_index=True,right_index=True)
        sample_df=pd.DataFrame()
        for i in samples:
            sample=pd.DataFrame(pos_class.ix[i]).transpose()
            samplex=pd.DataFrame(xtrain.ix[i,:]).transpose()
            conlistx=(new_xtrain,samplex)
            new_xtrain=pd.concat(conlistx)
            conlist=[new_ytrain,sample]
            new_ytrain=pd.concat(conlist)
            sample_entry=pd.merge(sample,samplex,left_index=True,right_index=True)
            samcon=[sample_df,sample_entry]
            sample_df=pd.concat(samcon)
        final_con=[new_df,sample_df]
        upsampled_df=pd.concat(final_con)
    return upsampled_df

#dummy variable where italian and mexican are one and every other cuisines are 0
y_train2.value_counts()
y_train_df=pd.DataFrame(y_train2)
dummies=pd.DataFrame(y_train_df['cuisinedummy'])
 
 y_train_df['cuisinedummy']=np.where(np.logical_or(y_train_df['cuisine']=='italian', y_train_df['cuisine']=='mexican'),1,0)

test=pd.DataFrame(dnsmpl(X_train2,dummies,.5))

test=pd.DataFrame(upsample(X_train2,dummies,.8))

dummies['cuisinedummy'].value_counts()
=======
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
>>>>>>> origin/master
>>>>>>> da51ee7859b1164963918ba8b9d6dd477cd07e0e
