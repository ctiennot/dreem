"""On this file you will implement you quality predictor function."""
# This file requires the functions.py file
from easy_import import extract_signals
import numpy as np
import functions as fun


def predict_quality(record):
    """Predict the quality of the signal.

    Input
    =====
    record: path to a record

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
    
    # Load pickle model (Random Forest here)
    rf = fun.loadModel("RF_benchmark")
    
    # predict classes
    print "Predicting..."
    y_pred = [rf.predict(X) for X in X_channels]
    
    # duplicate predictions and reshape
    y_pred = [np.tile(y_pred[i],(500,1)).T.reshape((-1)) for i in range(4)]
    
    # fill in the last measurements with missing values
    na_array = np.ndarray(raws.shape[1] - y_pred[0].shape[0])*np.nan
    y_pred = [np.hstack((y_pred[i],na_array)) for i in range(4)]
    
    # stack the 4 channels predictions and return them
    return np.vstack((y_pred[i] for i in range(4)))


if __name__ == "__main__":
    record = "../data/record1.h5"
    results = predict_quality(record)