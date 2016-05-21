# -*- coding: utf-8 -*-
"""
Benchmark with a simple Random Forest classifier on a data subset
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

# The RF classifier with 100 trees (8 jobs for multi-core)
rf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=8)
rf = rf.fit(X[train,:], y[train])

# Predictions (labels) ands probabilities for the test
preds_rf = rf.predict(X[test,:])
probs_rf = rf.predict_proba(X[test,:])

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

## Export model using pickle
#fun.saveModel(rf, "RF_benchmark")