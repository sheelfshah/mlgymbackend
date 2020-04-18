import numpy as np
import numpy.matlib
import pandas as pd

#assuming db is preprocessed, m rows, n columns(n inclusive of bias)
#last column of db contains result for training
def perceptron(db):
    train_db=db.sample(frac=1, random_state=0)
    X_train=train_db.iloc[:,:-1]#fill
    y_train=train_db.iloc[:,-1]#fill

    X_train_mat=X_train.to_numpy()
    y_train_mat=y_train.to_numpy()
    m=X_train_mat.shape[0]
    n=X_train_mat.shape[1]
    #initializations
    theta=2*(np.matlib.rand(n,1)-0.5)
    r=0.1 #vary and check
    max_iterations=100
    convergence_flag=True
    
    for j in range(max_iterations):
        convergence_flag=True
        for i in range(m):
            x=X_train_mat[i,:]
            x=x.reshape(1,n)
            z=np.matmul(x,theta)
            if z>=0:
                a=1
            else:
                a=0
            if not a==y_train_mat[i]:
                convergence_flag=False
                if a==0:
                    theta=theta+r*x.transpose()
                else:
                    theta=theta-r*x.transpose()
        if convergence_flag:
            break

    if convergence_flag:
        theta_string = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in theta)
    else:
        theta_string=""
    
    return theta_string

def perceptron_predict(db, theta):
    X=db.to_numpy()
    a=np.matmul(X,theta)
    pred=a
    for i in range(a.shape[0]):
        if pred[i,0]>=0:
            pred[i,0]=1
        else:
            pred[i,0]=0

    result_string = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in pred)
    return result_string