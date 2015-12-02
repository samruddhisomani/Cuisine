# -*- coding: utf-8 -*-

execfile('Original.py')
execfile('tfidf.py')

from sklearn.tree import DecisionTreeClassifier

s=DecisionTreeClassifier(max_depth=2, random_state=5)
s.fit(x,cuisine)
leaves=pd.Series(s.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves],axis=1)
m=list(leaves.value_counts().index.values)
for y in m:
    print y
    print idk[leaves==y]['cuisine'].value_counts()
leaves.value_counts()


s2=DecisionTreeClassifier(max_depth=3, random_state=5)
s2.fit(x,cuisine)
leaves2=pd.Series(s2.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves2],axis=1)
m=list(leaves2.value_counts().index.values)
for y in m:
    print y
    print idk[leaves2==y]['cuisine'].value_counts()
leaves2.value_counts()


s3=DecisionTreeClassifier(max_leaf_nodes=8, random_state=5,criterion='entropy')
s3.fit(x,cuisine)
leaves3=pd.Series(s3.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves3],axis=1)
m=list(leaves3.value_counts().index.values)
for y in m:
    print y
    print idk[leaves3==y]['cuisine'].value_counts()
leaves3.value_counts()

###############################################

s=DecisionTreeClassifier(max_depth=2, random_state=5, criterion='entropy')
s.fit(x,cuisine)
leaves=pd.Series(s.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves],axis=1)
m=list(leaves.value_counts().index.values)
for y in m:
    print y
    print idk[leaves==y]['cuisine'].value_counts()
leaves.value_counts()


s2=DecisionTreeClassifier(max_depth=3, random_state=5,criterion='entropy')
s2.fit(x,cuisine)
leaves2=pd.Series(s2.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves2],axis=1)
m=list(leaves2.value_counts().index.values)
for y in m:
    print y
    print idk[leaves2==y]['cuisine'].value_counts()
leaves2.value_counts()

s3=DecisionTreeClassifier(max_depth=4, random_state=5)
s3.fit(x,cuisine)
leaves3=pd.Series(s3.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves3],axis=1)
m=list(leaves3.value_counts().index.values)
for y in m:
    print y
    print idk[leaves3==y]['cuisine'].value_counts()
leaves3.value_counts()

#####################

s=DecisionTreeClassifier(max_leaf_nodes=4, random_state=5,class_weight='balanced')
s.fit(x,cuisine)
leaves=pd.Series(s.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves],axis=1)
m=list(leaves.value_counts().index.values)
for y in m:
    print y
    print idk[leaves==y]['cuisine'].value_counts()
leaves.value_counts()


s2=DecisionTreeClassifier(max_depth=3, random_state=5)
s2.fit(x,cuisine)
leaves2=pd.Series(s2.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves2],axis=1)
m=list(leaves2.value_counts().index.values)
for y in m:
    print y
    print idk[leaves2==y]['cuisine'].value_counts()
leaves2.value_counts()

s3=DecisionTreeClassifier(max_depth=4, random_state=5)
s3.fit(x,cuisine)
leaves3=pd.Series(s3.apply(x),name='leaves')
idk=pd.concat([cuisine,leaves3],axis=1)
m=list(leaves3.value_counts().index.values)
for y in m:
    print y
    print idk[leaves3==y]['cuisine'].value_counts()
leaves3.value_counts()
