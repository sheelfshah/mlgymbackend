import numpy as np
import pandas as pd

def LinearRegression(X,Y,LearningRate=0.01,Termination=0,max_iters=100,Reg=0):
    X=X.to_numpy()
    Y=Y.to_numpy()
    m,n=X.shape
    Y=Y.reshape(m,1)
    theta=np.zeros(n)
    theta=theta.reshape(n,1)
    cost=10**5
    num_iters=0
    while(cost>Termination and num_iters<max_iters):
        temp=X.dot(theta)-Y
        hx=(LearningRate*(X.T).dot(temp))/m
        hx=hx.reshape(n,1)
        theta[0,0]=theta[0,0]-hx[0,0]
        theta[1:]=theta[1:]*(1-Reg/m)-hx[1:]
        cost_prev=cost

        cost=((temp.T).dot(temp)+Reg*(theta.T).dot(theta))/(2*m)
        if(cost_prev-cost<Termination):
            return theta.reshape(n,1)
        if(cost>cost_prev):
            LearningRate=LearningRate/2
        num_iters+=1
    return theta.reshape(n,1)

def linreg_predict(X_train,theta):
    X_train=X_train.to_numpy()
    m,n=X_train.shape
    X_train=np.array(X_train)
    theta=theta.reshape(n,1)
    return X_train.dot(theta)






