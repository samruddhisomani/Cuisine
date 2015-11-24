from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot
from itertools import chain


train=pd.read_json("../Data/train.json")

cuisine=train['cuisine']

##bar plot over cuisine
cuisine_vc=cuisine.value_counts()
cuisine.plot()

fig=cuisine_vc.plot(kind='bar',title='Number of Recipes by Cuisine').get_figure()
fig.savefig('cuisinecount.svg')

ingredients=list(train['ingredients'])
ingredients_f=list(chain.from_iterable(ingredients))
IF=pd.Series(ingredients_f)
IFVC=IF.value_counts()

##bar plot over ingredients
fig=IFVC.head(20).plot(kind='bar', title='Top Twenty Ingredients').get_figure()
fig.savefig('../Images/ingredientcount.svg')
