from __future__ import unicode_literals
import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from itertools import chain


train=pd.read_json("../Data/train.json")

cuisine_vc=train['cuisine'].value_counts()

##bar plot over cuisine
f=cuisine_vc.plot(kind='bar',title='Number of Recipes by Cuisine').get_figure()#this is figure
f.tight_layout()
f.set_size_inches(8,6)
f.savefig('../Images/cuisinecount_e.png',bbox_inches='tight',dpi=100)


ingredients=list(train['ingredients'])
ingredients_f=list(chain.from_iterable(ingredients))
IF=pd.Series(ingredients_f)
IFVC=IF.value_counts()

##bar plot over ingredients
f=IFVC.head(20).plot(kind='bar', title='Top Twenty Ingredients').get_figure()
f.tight_layout()
f.set_size_inches(8,7)
f.savefig('../Images/ingredientcount.png',bbox_inches='tight',dpi=100)
