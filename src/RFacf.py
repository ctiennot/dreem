# -*- coding: utf-8 -*-
"""
Random Forest on top of ACF
"""
import functions as fun
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

###################### Copy from incrementsVarRF.py ###########################
# Retrieve X and y matrices from files
X, y = fun.load_quality_dataset()
# Generate train and test sets
train, test = fun.getTrainTest(X, y, subset=1)

# Computing piece-wise and overall variances for raw and filtered signals
var_features_filtered = fun.extract_piecewise_var(X[:,500:1000], w_size=10)
var_filtered = fun.extract_piecewise_var(X[:,500:1000], w_size=10)      
var_features_raw = fun.extract_piecewise_var(X[:,0:500], w_size=10)
var_raw = fun.extract_piecewise_var(X[:,0:500], w_size=10)   

###################### Copy from incrementsVarRF.py ###########################                

# computing acf for filtered and raw signal
intACF_filtered = np.array(map(fun.acf, np.diff(X[:,500:1000])))
intACF_raw = np.array(map(fun.acf, np.diff(X[:,0:500])))

# Stacking all the features (50 + 1 + 50 + 1)
final_features = np.hstack((var_features_filtered, var_filtered,
                            var_features_raw, var_raw, intACF_filtered,
                            intACF_raw))

# Fitting a Random Forest on top
rf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=8)
rf = rf.fit(final_features[train], y[train])

# Predictions (labels) ands probabilities for the test
preds_rf = rf.predict(final_features[test,:])
probs_rf = rf.predict_proba(final_features[test,:])

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
# fun.saveModel(rf, "RF_acf")