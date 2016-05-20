# -*- coding: utf-8 -*-
"""
Feature engineering using increments anad their piece-wise variances
& Random Forest classifier on top of that
"""
# This file requires the functions.py file
import functions as fun
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Retrieve X and y matrices from files
X, y = fun.load_quality_dataset()
# Generate train and test sets
train, test = fun.getTrainTest(X, y, subset=1)

w_size = 10 # window size to design segments

# Tuples delimiting segments for the filtered signal
var_segments = zip(np.arange(500, 1000, w_size), 
                   np.arange(500, 1000+w_size, w_size) + w_size)
                   
# Computing variances on each of those segments
var_features = np.zeros((X.shape[0],len(var_segments)))
for i, (a,b) in enumerate(var_segments):
    var_features[:,i] = np.diff(X[:,range(a,b)]).var(axis=1)

# Adding the global variance as a last feature
var_features = np.hstack((var_features, 
                          np.diff(X[:,500:1000]).var(axis=1).reshape(-1,1)))

#######################################                
var_features_raw = np.zeros((X.shape[0],len(var_segments)))
for i, (a,b) in enumerate(var_segments):
    var_features_raw[:,i] = np.diff(X[:,range(a-500,b-500)]).var(axis=1)

# Adding the global variance as a last feature
var_features_raw = np.hstack((var_features_raw, 
                          np.diff(X[:,0:500]).var(axis=1).reshape(-1,1)))

var_features = np.hstack((var_features, var_features_raw))
#########################################
# Fitting a Random Forest on top
rf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=8)
rf = rf.fit(var_features[train], y[train])

# Predictions (labels) ands probabilities for the test
preds_rf = rf.predict(var_features[test,:])
probs_rf = rf.predict_proba(var_features[test,:])

# Confusion matrix
fun.plotCM(y[test], preds_rf)

# Accuracy
print 'Accuracy =', np.mean(y[test]==preds_rf)
print 'Multiclass log-loss =', log_loss(y[test], probs_rf)

# The importance of the features
rf_importance = rf.feature_importances_
plt.figure()
plt.bar(range(rf_importance.shape[0]), rf_importance)
plt.show()