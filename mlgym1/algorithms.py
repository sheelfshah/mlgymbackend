import numpy as np
import numpy.matlib
import pandas as pd
import numpy.random as rand

#cuz django loves strings
def numpy_to_str(theta):
    return('\n'.join('\t'.join('%0.3f' %x for x in y) for y in theta))

def str_to_numpy(theta_string):
    return(np.array([[float(j) for j in i.split('\t')] for i in theta_string.splitlines()]))

#assuming db is preprocessed, m rows, n columns(n inclusive of bias)
#last column of db contains result for training
def perceptron(db):
    train_db=db.sample(frac=1, random_state=0)
    X_train=train_db.iloc[:,:-1]
    y_train=train_db.iloc[:,-1]

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
        theta_string = numpy_to_str(theta)
    else:
        theta_string=""
    
    return theta_string

#assuming db has same preprocessing as training db and no result column is present
def perceptron_predict(db, theta):
    X=db.to_numpy()
    a=np.matmul(X,theta)
    pred=a
    for i in range(a.shape[0]):
        if pred[i,0]>=0:
            pred[i,0]=1
        else:
            pred[i,0]=0
    return pred

#both numpy
def perceptron_accuracy(pred,actual):
    true_pos=0
    true_neg=0
    false_pos=0
    false_neg=0
    for i in range(pred.shape[0]):
        if pred[i,0]==actual[i,0]:
            if pred[i,0]==1:
                true_pos+=1
            else:
                true_neg+=1
        else:
            if pred[i,0]==1:
                false_pos+=1
            else:
                false_neg+=1
    prec=true_pos/(true_pos+false_pos)
    rec=true_pos/(true_pos+false_neg)
    f1=2*prec*rec/(prec+rec)
    acc=(true_pos+true_neg)/(pred.shape[0])
    return true_pos,false_pos,false_neg,true_neg,f1,acc