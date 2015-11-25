from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

execfile('tfidf.py')

param_grid={'max_depth':[2,4,6,8],'n_estimators':[10,100,500,1000]}

rf=RandomForestClassifier(warm_start=True, random_state=0,n_jobs=4,oob_score=True)

c=GridSearchCV(rf,param_grid,cv=2)

c.fit(x,cuisine)
c.best_estimator_
c.best_estimator_.oob_score_

param_grid={'max_depth':[8,15,25],'n_estimators':[100,500,1000]}
c.fit(x,cuisine)
c.best_estimator_
c.best_score_