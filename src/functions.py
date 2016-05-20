# -*- coding: utf-8 -*-
"""
This file contains various functions to be called from the script files
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# Plot a confusion matrix given labels and predictions
def plotCM(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def load_quality_dataset(ratio=0.5, valid=0., seed=1789):
    '''
    Load the data from quality_dataset.h5 and split them in train/val/test sets
    ---
    Returns a tuple with (X, y) with np.arrays
    '''
    # load the .h5 file
    quality = h5py.File("../data/quality_dataset.h5", "r") # reading buffer
    X = quality['dataset'][:] # load in memory
    quality.close() # close buffer
    
    y = X[:,1000] # retrieve labels fom last column
    X = X[:,0:1000] # remove last columns from 
    
    return X, y
    
def getTrainTest(X, y, ratio=0.8, valid=0., seed=1789, subset = 1.):
    '''
    Generate np.arrays with indices to split data in train/val/test sets
    ---
    X : the feature matrix
    y : the labels matrix
    ratio : the proportion of observations to assign to the train set
    val : proportion of the train to keep for validation
    seed : seed when splitting
    subset : draw only a subset of the data, float between 0 and 1
    '''
    train = []
    test = []
    val = []
    
    n = X.shape[0] # number of observations
    
    # Set the seed right
    if seed!=0:
        np.random.seed(seed)

    # If a subset is require we draw it from the data random
    index = np.random.choice(range(n), subset*n, False)
    
    # we shuffle the index (inplace)
    np.random.shuffle(index)
    
    # Now create train, test and validation if required
    train = index[range(int(ratio*(1-valid)*subset*n))]
    val = index[range(int(ratio*(1-valid)*subset*n), int(ratio*subset*n))]
    test = index[range(int(ratio*subset*n), int(subset*n))]
    
    # If no validation return only train and test
    if valid==0.:
        return np.array(train), np.array(test)
    else:
        return np.array(train), np.array(test), np.array(val)
        
def extract_piecewise_var(X, w_size = 10):
    '''
    Take increments of the time series and Compute piece-wise variances of 
    the increments (segments of length 10 by default)
    ---
    X : the feature matrix
    ratio : the proportion of observations to assign to the train set
    val : proportion of the train to keep for validation
    seed : seed when splitting
    subset : draw only a subset of the data, float between 0 and 1
    '''
    n = X.shape[0] # number of observations
    p = X.shape[1] # number of features/times
        
    # Tuples delimiting segments for the filtered signal
    var_segments = zip(np.arange(0, p, w_size), 
                       np.arange(0, p+w_size, w_size) + w_size)
                       
    # Computing variances on each of those segments
    var_features = np.zeros((n,len(var_segments)))
    for i, (a,b) in enumerate(var_segments):
        var_features[:,i] = np.diff(X[:,range(a,b)]).var(axis=1)
    
    return var_features
