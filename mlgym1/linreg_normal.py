import numpy as np
import pandas as pd

#x: MxN
#y: Mx1
def linreg_normal_train(x,y):
    x=x.to_numpy()
    y=y.to_numpy()
    m,n=x.shape
    x=np.c_[np.ones(m),x]
    params=np.linalg.pinv((x.T)@x)@x.T@y
    return (params.reshape(params.shape[0],1))

def linreg_normal_predict(x,params):
    x=x.to_numpy()
    m,n=x.shape
    x=np.c_[np.ones(m),x]
    y=x@params
    return (y.reshape(y.shape[0],1))
