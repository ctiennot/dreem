"""On this file you will implement you quality predictor function."""
# This file requires the functions.py file
from easy_import import extract_signals
import numpy as np
import functions as fun
import matplotlib.pyplot as plt


def predict_quality(record, model="RF_variances", filtered_signal = False):
    """Predict the quality of the signal.

    Input
    =====
    record: path to a record
    model: pickle model to use
    filtered_signal: if true then return the input signal array as the second
    element of a tuple

    Output
    ======
    results: a list of 4 signals between 0 and 1 estimating the quality
    of the 4 channels of the record at each timestep.
    This results must have the same size as the channels.
    """
    # Extract signals from record
    raws, filtered = extract_signals(record)
    
    # divide 4 signals in 4 design matrix (1 rows = 2s window = 500 features)
    # (generatorto avoid storing matrices)
    X_channels = (np.hstack((fun.extract_windows(raws[i,:]),
                  fun.extract_windows(filtered[i,:]))) 
                  for i in range(4))
                      
    # quick function to extract features (piece-wise variances)
    def create_X(X):
        '''quick function to extract piece-wise variance functions'''
        # Computing piece-wise and overall variances for raw and filtered
        # signals
        var_features_filtered = fun.extract_piecewise_var(X[:,500:1000], w_size=10)
        var_filtered = fun.extract_piecewise_var(X[:,500:1000], w_size=500)      
        var_features_raw = fun.extract_piecewise_var(X[:,0:500], w_size=10)
        var_raw = fun.extract_piecewise_var(X[:,0:500], w_size=500)   

        # Stacking all the features (50 + 1 + 50 + 1)
        var_features = np.hstack((var_features_filtered, var_filtered, 
                                  var_features_raw, var_raw))
        return var_features
    
    # Load pickle model (Random Forest here)
    rf = fun.loadModel(model)

    # predict classes
    print "Predicting..."
    
    # Apply preprocessing according to the model chosen
    if model=="RF_variances":
        y_pred = [rf.predict(create_X(X)) for X in X_channels]
    else:
        y_pred = [rf.predict(X) for X in X_channels]
    
    # duplicate predictions and reshape
    y_pred = [np.tile(y_pred[i],(500,1)).T.reshape((-1)) for i in range(4)]
    
    # fill in the last measurements with missing values
    na_array = np.ndarray(raws.shape[1] - y_pred[0].shape[0])*np.nan
    y_pred = [np.hstack((y_pred[i],na_array)) for i in range(4)]
    
    # stack the 4 channels predictions and return them
    if filtered_signal:
        return np.vstack((y_pred[i] for i in range(4))), filtered
        
    return np.vstack((y_pred[i] for i in range(4)))


if __name__ == "__main__":
    record = "../data/record1.h5"
    results, filtered = predict_quality(record, filtered_signal=True)
    
visualize = True # plot some predictions

if __name__ == "__main__" and visualize:
    subset = np.arange(100000, 1000000,1) # time range
    ch = range(4) # channels to plot
    
    f, axs = plt.subplots(len(ch),1)
    f.tight_layout()
    for i in ch:
        axs[i].plot(subset, filtered[i,subset]) #signal
        axs[i].plot(subset[np.where(results[i,subset]==1)[0]], 
             filtered[i,subset[results[i,subset]==1]], 'r-')
        axs[i].set_title("Channel "+str(i))
        axs[i].axes.get_xaxis().set_visible(False)
        axs[i].axes.get_yaxis().set_visible(False)
        