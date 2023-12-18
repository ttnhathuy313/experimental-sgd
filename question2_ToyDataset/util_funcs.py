import math
import time

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# Most of the code related to toy dataset and simple fully connected (FC) network
# can be found at the main blog: https://machinelearningcoban.com/2017/02/24/mlp/
# and at the GitHub page: https://github.com/tiepvupsu/ebookMLCB

# The part on SAGA is a simple implementation and can be further optimized
# Base on the flow similar to what is mentioned at: 
# https://en.wikipedia.org/wiki/Stochastic_variance_reduction
# and https://github.com/kilianFatras/variance_reduced_neural_networks


def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each row of Z is a set of scores.
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A


def crossentropy_loss(Yhat, y):
    """
    Yhat: a numpy array of shape (Npoints, nClasses) -- predicted output 
    y: a numpy array of shape (Npoints) -- ground truth. We don't need to use 
    the one-hot vector here since most of the elements are zeros. When programming
    in numpy, we need to use the corresponding indexes only.
    """
    id0 = range(Yhat.shape[0])
    return -np.mean(np.log(Yhat[id0, y]))


def mlp_init(d0, d1, d2):
    """ 
    Initialize W1, b1, W2, b2 
    d0: dimension of input data 
    d1: number of hidden unit 
    d2: number of output unit = number of classes
    """
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros(d1)
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros(d2)
    return (W1, b1, W2, b2)


def mlp_predict(X, W1, b1, W2, b2):
    """
    Suppose that the network has been trained, predict class of new points. 
    X: data matrix, each ROW is one data point.
    W1, b1, W2, b2: learned weight matrices and biases 
    """
    Z1 = X.dot(W1) + b1    # shape (N, d1)
    A1 = np.maximum(Z1, 0) # shape (N, d1)
    Z2 = A1.dot(W2) + b2   # shape (N, d2)
    return np.argmax(Z2, axis=1)


def mlp_forward(X, W1, b1, W2, b2):

    # feedforward 
    Z1 = X.dot(W1) + b1       # shape (N, d1)
    A1 = np.maximum(Z1, 0)    # shape (N, d1)
    Z2 = A1.dot(W2) + b2      # shape (N, d2)
    Yhat = softmax_stable(Z2) # shape (N, d2)

    return Z1, A1, Z2, Yhat


def mlp_backward(X, y, Z1, A1, Yhat):
    # back propagation
    id0 = range(Yhat.shape[0])
    Yhat[id0, y] -=1 
    E2 = Yhat / X.shape[0]                # shape (N, d2)
    dW2 = np.dot(A1.T, E2)     # shape (d1, d2)
    db2 = np.sum(E2, axis = 0) # shape (d2,)
    E1 = np.dot(E2, W2.T)      # shape (N, d1)
    E1[Z1 <= 0] = 0            # gradient of ReLU, shape (N, d1)
    dW1 = np.dot(X.T, E1)      # shape (d0, d1)
    db1 = np.sum(E1, axis = 0) # shape (d1,)
    
    return dW1, db1, dW2, db2


def mlp_fit(X, y, W1, b1, W2, b2, eta, batchSize = 1, nIter = 10000):
    loss_hist = []
    
    for i in range(nIter):
        
        sampInd = np.random.choice(X.shape[0], (batchSize, ), replace = False)
        XFeed = X[sampInd, :]
        yFeed = y[sampInd]
        
        # feedforward 
        Z1, A1, Z2, Yhat = mlp_forward(XFeed, W1, b1, W2, b2)
        
        loss = crossentropy_loss(mlp_forward(X, W1, b1, W2, b2)[3], y)
        loss_hist.append(loss)
        if i % 1000 == 0: # print loss after each 1000 iterations
            print("iter %d, loss: %f" %(i, loss))
            
        # back propagation
        dW1, db1, dW2, db2 = mlp_backward(XFeed, yFeed, Z1, A1, Yhat)

        # Gradient Descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
    return (W1, b1, W2, b2, loss_hist)


def mlp_gradTest(X, y, W1, b1, W2, b2, batchSize = 1, nIter = 10000):
    
    gradHistory = {
        'dW1': [],
        'db1': [],
        'dW2': [],
        'db2': [],
    }
    
    for i in range(nIter):
        
        sampInd = np.random.choice(X.shape[0], (batchSize, ), replace = False)
        XFeed = X[sampInd, :]
        yFeed = y[sampInd]
        
        # feedforward 
        Z1, A1, Z2, Yhat = mlp_forward(XFeed, W1, b1, W2, b2)
            
        # back propagation
        dW1, db1, dW2, db2 = mlp_backward(XFeed, yFeed, Z1, A1, Yhat)
        
        gradHistory['dW1'].append(dW1)
        gradHistory['db1'].append(db1)
        gradHistory['dW2'].append(dW2)
        gradHistory['db2'].append(db2)
    
    gradHistory['dW1'] = np.array(gradHistory['dW1'])
    gradHistory['db1'] = np.array(gradHistory['db1'])
    gradHistory['dW2'] = np.array(gradHistory['dW2'])
    gradHistory['db2'] = np.array(gradHistory['db2'])
    
    return gradHistory


def mlpSAGA(X, y, W1, b1, W2, b2, eta, nIter = 10000):
    
    loss_hist = []
    cache = createCacheSAGA(X, y, W1, b1, W2, b2)
    
    for i in range(nIter):
        
        sampInd = np.random.choice(X.shape[0], (1, ), replace = False)
        XFeed = X[sampInd, :]
        yFeed = y[sampInd]
        
        # feedforward 
        Z1, A1, Z2, Yhat = mlp_forward(XFeed, W1, b1, W2, b2)
        
        loss = crossentropy_loss(mlp_forward(X, W1, b1, W2, b2)[3], y)
        loss_hist.append(loss)
        if i % 1000 == 0: # print loss after each 1000 iterations
            print("iter %d, loss: %f" %(i, loss))
            
        # back propagation
        dW1, db1, dW2, db2 = mlp_backward(XFeed, yFeed, Z1, A1, Yhat)

        # Gradient Descent update
        W1 += -eta * (dW1 - cache['W1'][sampInd, :, :][0] + np.mean(cache['W1'], axis = 0))
        b1 += -eta * (db1 - cache['b1'][sampInd, :][0] + np.mean(cache['b1'], axis = 0))
        W2 += -eta * (dW2 - cache['W2'][sampInd, :, :][0] + np.mean(cache['W2'], axis = 0))
        b2 += -eta * (db2 - cache['b2'][sampInd, :][0] + np.mean(cache['b2'], axis = 0))
        
        cache['W1'][sampInd, :, :] = dW1
        cache['b1'][sampInd, :] = db1
        cache['W2'][sampInd, :, :] = dW2
        cache['b2'][sampInd, :] = db2
        
    return (W1, b1, W2, b2, loss_hist)


def createCacheSAGA(X, y, W1, b1, W2, b2):
    
    cache = {
        'W1': np.zeros((X.shape[0], W1.shape[0], W1.shape[1])),
        'b1': np.zeros((X.shape[0], b1.shape[0])),
        'W2': np.zeros((X.shape[0], W2.shape[0], W2.shape[1])),
        'b2': np.zeros((X.shape[0], b2.shape[0])),
    }
    
    for i in range(X.shape[0]):
        
        sampInd = np.array([i])
        
        # feedforward 
        Z1, A1, Z2, Yhat = mlp_forward(X[sampInd, :], W1, b1, W2, b2)
        
        # back propagation
        dW1, db1, dW2, db2 = mlp_backward(X[sampInd, :], y[i], Z1, A1, Yhat)
        
        cache['W1'][sampInd, :, :] = dW1
        cache['b1'][sampInd, :] = db1
        cache['W2'][sampInd, :, :] = dW2
        cache['b2'][sampInd, :] = db2
    
    return cache


def relativeError(a, b):
    return np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))




