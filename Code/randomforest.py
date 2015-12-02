from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import sam

execfile('tfidf.py')

X = x.toarray()

param_grid={'max_depth':[2,4,6,8],'n_estimators':[10,100,500,1000]}

rf=RandomForestClassifier(warm_start=True, random_state=0,n_jobs=-1,oob_score=True)

c=GridSearchCV(rf,param_grid,cv=2)



c.fit(X,cuisine)
c.best_estimator_    #depth = 8, trees = 1000
c.best_estimator_.oob_score_      #0.42744


rf2=RandomForestClassifier(max_depth = 25, n_estimators = 1000,warm_start=True, random_state=0,n_jobs=-1)
rf2.fit(X,cuisine)

rf_test = v.transform(test["ingredients"])
rf_test_array = rf_test.toarray()
rf2_pred = rf2.predict(rf_test_array)
results = pd.DataFrame()
results["id"] = test["id"]
results["cuisine"] = rf2_pred 



param_grid={'max_depth':[8,15,25],'n_estimators':[100,500,1000]}
c.fit(x,cuisine)
c.best_estimator_   #depth = 25, trees = 1000      ~55% accuracy
c.best_score_        #0.5555



#heatmap
rf2.fit(X_train,y_train)
rf_hm_pred = rf2.predict(x)
rf_hm_df = ConfusionMatrix(cuisine,rf_hm_pred,"precision")
rf_cm = confusion_matrix(y_test.as_matrix(),rf_hm_pred,labels=rf_hm_df.index)
heatmap(rf_cm,xticklabels=rf_hm_df.index,yticklabels=rf_hm_df.index)
heatmap(rf_hm_df)
hmwrapper(rf_hm_df,'../Images/rf_heatmap.png')