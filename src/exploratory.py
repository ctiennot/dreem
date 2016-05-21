# -*- coding: utf-8 -*-
"""
This file gives some insights about the data, it is an exploratory analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import functions as fun

np.random.seed(seed=1789)

# Load data
X, y = fun.load_quality_dataset()

pos = np.array(range(y.shape[0]))[y==1.] # good signal index
neg = np.array(range(y.shape[0]))[y==0.] # bad signal index

# Look at random trajectories in the data
plt.figure()
axes = plt.gca()
axes.set_ylim([-500,500])
for i in np.random.choice(pos, 300, False):
    plt.plot(range(500), X[i,500:1000], color=(.8,.8,.8))
for i in np.random.choice(neg, 300, False):
    plt.plot(range(500), X[i,500:1000], color=(1.,0.,0.,.6))
plt.title("Random trajectories\n")
plt.xlabel("Time (1 unit = 4 ms)")
plt.ylabel("EEG (mV)")
plt.show()

# The max values for each line
max_v = X[:,500:1000].max(axis=1)
min_v = X[:,500:1000].min(axis=1)
amp_v = max_v-min_v
var_v = X[:,500:1000].var(axis=1)
mean_v = X[:,500:1000].mean(axis=1)

# Distribution of log-amplitudes
plt.figure()
plt.hist(np.log(amp_v[pos]), bins=50, color=(.8,.8,.8,1.))
plt.hist(np.log(amp_v[neg]), bins=50, color=(1.,0.,0.,.6))
plt.xlabel("log(Amp(EEG))")
plt.ylabel("Count")
plt.title("Distribution of log-amplitudes\n")
plt.show()

# Distribution of log-variances
plt.figure()
plt.hist(np.log(var_v[pos]), bins=50, color=(.8,.8,.8,1.))
plt.hist(np.log(var_v[neg]), bins=50, color=(1.,0.,0.,.6))
plt.xlabel("log(Var(EEG))")
plt.ylabel("Count")
plt.title("Distribution of log-variances\n")
plt.show()

# Distribution of means
plt.figure()
plt.hist(mean_v[pos], bins=50, color=(.8,.8,.8,1.))
plt.hist(mean_v[neg], bins=50, color=(1.,0.,0.,.6))
plt.title("Distribution of means")

# Look at random trajectories in the data after taking increments
plt.figure()
axes = plt.gca()
axes.set_ylim([-500,500])
for i in np.random.choice(pos, 300, False):
    plt.plot(range(499), np.diff(X[i,500:1000]), color=(.8,.8,.8))
for i in np.random.choice(neg, 300, False):
    plt.plot(range(499), np.diff(X[i,500:1000]), color=(1.,0.,0.,.6))
plt.title("Random trajectories")
plt.xlabel("Time (1 unit = 4 ms)")
plt.ylabel("EEG (mV)")
plt.show()

# Distribution of log-variances for increments
var_increments = np.diff(X[:,500:1000]).var(axis=1)
plt.figure()
plt.hist(np.log(var_increments[pos]), bins=50, color=(.8,.8,.8,1.))
plt.hist(np.log(var_increments[neg]), bins=50, color=(1.,0.,0.,.6))
plt.title("Distribution of log-variances")