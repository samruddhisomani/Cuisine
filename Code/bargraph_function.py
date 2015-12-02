#to get bargraph call the bargraph function with preds as your predictions and ytest as the actuals
def counter(array):
    df=pd.DataFrame(array)
    df.columns=['name']
    return df['name'].value_counts()
    
def bargraph(preds,ytest):
    pred_counts=counter(preds)
    y_vc=ytest.value_counts()
    y_vc=pd.DataFrame(y_vc)
    y_vc.columns=['actual']
    y_vc['predicted']=pred_counts
    return y_vc.plot(kind='bar')
    
    
