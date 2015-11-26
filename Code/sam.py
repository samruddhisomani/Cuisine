import pandas as pd
from seaborn import heatmap

def balanced_samples(number,classes,dfO,columns,group):
    df=pd.DataFrame(columns=columns)
    for q in classes:
        grouped=dfO[dfO[group]==q]
        answer=grouped.sample(number, replace=True)
        df=df.append(answer)
    return df

def ConfusionMatrix(y_true,y_pred,kind):

    y_t=pd.Series(y_true,name='Truth')
    y_p=pd.Series(y_pred,name='Predictions')
    df=pd.crosstab(index=y_t, columns=y_p).fillna(0)

    if kind=='original':
        df_new=df
    elif kind=='precision':
        #how many of selected items are relevant:
        #divide over sum of columns
        df_new=df.div(df.sum(axis=1)).fillna(0)
    elif kind=='recall':
        #how many relevant items are selected:
        #divide over sum of rows
        df_new=df.div(df.sum(axis=0)).fillna(0)
    
    return df_new
    
    
    
