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
    df=pd.crosstab(index=y_t, columns=y_p,dropna=False).fillna(0)

    if kind=='original':
        df_new=df
    elif kind=='precision':
        #how many of selected items are relevant:
        #divide row over sum of columns
        df_new=df.div(df.sum(axis=1),axis='index').fillna(0)
    elif kind=='recall':
        #how many relevant items are selected:
        #divide each column over sum of rows
        df_new=df.div(df.sum(axis=0),axis='columns').fillna(0)
    return df_new
    
def hmwrapper(cm,filename):
    h=heatmap(cm).get_figure()
    ax=h.add_subplot(111)
    ax.set_xlabel('Predictions')
    h.tight_layout()
    h.set_size_inches(8,5.5)
    h.savefig(filename,bbox_inches='tight',dpi=100)

def SubFile(ids,answers,filename):
    sub=pd.concat([ids,answers],axis=1).rename(columns={0:'cuisine'})
    sub.to_csv(filename,index=False,encoding='utf-8')
    print "saved to", filename


    
