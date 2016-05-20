"""On this file you will implement you quality predictor function."""
from easy_import import extract_signals
import numpy as np


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
    # Initialize results (same size as the channels)
    results = np.zeros((raws.shape))

    return results


if __name__ == "__main__":
    record = "record1.h5"
    results = predict_quality(record)