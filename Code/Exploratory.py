import pandas as pd
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
from __future__ import unicode_literals
import matplotlib.pyplot

%pylab inline

train=pd.read_json("../Data/train.json")

cuisine=train['cuisine']

cuisine_vc=cuisine.value_counts()
cuisine.plt()

cuisine_vc.plot(kind='bar',title='Number of Recipes by Cuisine')
