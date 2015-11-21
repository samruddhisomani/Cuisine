  from pandas_confusion import ConfusionMatrix

  cm=ConfusionMatrix(y_test,pred)

  cm.plot(cmap=c.get_cmap('PuBu'))
