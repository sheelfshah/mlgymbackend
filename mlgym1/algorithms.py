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
   
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def back_prop_error(prev_error, curr_layer, matrix_from_curr_to_prev):
    return np.multiply(np.multiply(np.matmul(matrix_from_curr_to_prev,prev_error.transpose()).transpose(),curr_layer),1-curr_layer)

def feed_forward(X,thetas):
    m=X.shape[0]
    z1=np.matmul(np.c_[np.ones(m),X],thetas[0])
    list_of_a=[sigmoid(z1)]
    for theta in thetas[1:]:
        z_i=np.matmul(np.c_[np.ones(m),list_of_a[-1]],theta)
        list_of_a.append(sigmoid(z_i))
    return list_of_a

def get_errors(list_of_a, y_train,thetas):
    last_error=list_of_a[-1]-y_train
    deltas=[]
    deltas.append(last_error)
    i=-2
    while i>=-len(list_of_a):
        to_use_theta=np.delete(thetas[i+1],(0),axis=0)
        last_error=back_prop_error(last_error,list_of_a[i],to_use_theta) 
        deltas.append(last_error)
        i-=1
    return deltas[::-1]

def get_grads(errors, list_of_a, X):
    grads=[]
    new_loa=[X]+list_of_a[:-1]
    for i in range(len(new_loa)):
        grad_i=np.matmul(errors[i].transpose(),np.c_[np.ones(new_loa[i].shape[0]),new_loa[i]])
        grads.append(grad_i.transpose())
    return grads

#assumes preformatted training data in numpy format
#x: MxN
#y: MxC; c is the number of classes; y is formatted ideally
def nn4(X_train, y_train,alpha=0.5,lamda=0,num_epochs=300,neurons=10):
    num_neurons=[X_train.shape[1],neurons+30,neurons,y_train.shape[1]]
    thetas=[]

    # get random thetas
    for i in range(1,len(num_neurons)):
        theta_i=rand.random_sample((num_neurons[i-1]+1,num_neurons[i]))-0.5
        thetas.append(theta_i)
    thets=thetas
    m=X_train.shape[0]
    for zz in range(num_epochs):
        loa=feed_forward(X_train, thets)
        err=get_errors(loa, y_train, thets)
        grads=get_grads(err, loa, X_train)
        for i in range(len(thets)):
            temp_theta=thets[i]
            temp_theta[0,:]=0
            thets[i]=thets[i]-((alpha*grads[i])/m)-((lamda*temp_theta)/m)
    return thets

def db_to_nn4(x_db,y_db):
    x_mat=x_db.to_numpy()
    y_mat=np.zeros(y_db.shape[0]*10).reshape(y_db.shape[0],10)  #hardcoded temporarily
    for i in range(y_mat.shape[0]):
        y_mat[i,y_db.iloc[i]]=1
    return(nn4(x_mat,y_mat))

def nn4_predict(db,theta_list):
    x=db.to_numpy()
    pred=feed_forward(x,theta_list)[-1]
    results=np.argmax(pred,axis=1)
    results=results.reshape(results.shape[0],1)
    print (results.shape)
    return results