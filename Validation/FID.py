## FID is a method that calculates the distance between distribution of datapoints.
## Since out data is high dimensional, for each time step, one feature of each trajectory will be taking
## This will result in distributions of real and synthetic data, one for each datapoint and feature
## Finally, the FID is calculated for each of these distributions, obtaining a FID score in time for each feature

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

## Available routes
## EHAM_LIMC
## ESSA_LFPG
## LOWW_EGLL

## ESSA_LFPG_2 (filtered for going up trajectories)

## Comparison options
## val
## train
## test

## Import the data EDIT THE ROUTE
route = "EHAM_LIMC"
n = 40
filtered = True
compared = "test"
L = 10

direction_compared = "C:/Sofia/TU Delft/Bsc2/Proyect/second semester/" + str(route) + "_" + compared + "_data_n=" + str(n)
direction_gen = "C:/Sofia/TU Delft/Bsc2/Proyect/second semester/" + str(route) + "_gen_data_n=" + str(n)

if L != 0:
    direction_gen = direction_gen + '_L=' + str(L)

if filtered:
    direction_compared = direction_compared + '_filtered.npz'
    direction_gen = direction_gen + '_filtered.npz'
else:
    direction_compared = direction_compared + '.npz'
    direction_gen = direction_gen + '.npz'

r_data_npz = np.load(direction_compared)
s_data_npz = np.load(direction_gen)

key = r_data_npz.files[0]

r_data = r_data_npz[key]
s_data = s_data_npz['data']

r_data = [item for sublist in r_data for item in sublist]
s_data = [item for sublist in s_data for item in sublist]
## Stuff for validating the tool

def FID(real_data, synth_data):
    ## Clean the data to obtain an array with just the 4 relevant info
    #real_data = np.delete(real_data, 0, 0)
    #real_data = np.delete(real_data, np.s_[4:], 1)
    #synth_data = np.delete(synth_data, 0, 0)
    #synth_data = np.delete(synth_data, np.s_[4:], 1)
    ## look for start of trajectories
    real_index = []
    synth_index = []
    for i in range(len(real_data)):
        if real_data[i][3] == 0:
            real_index.append(i)
    for i in range(len(synth_data)):
        if synth_data[i][3] == 0:
            synth_index.append(i)
    flight_size = real_index[1] ## This assumes all trajectories have the same size, which they do in this experiment 
    # Calculate the global mean and standard deviation for normalization
    global_mean = np.mean(np.vstack((real_data, synth_data)), axis=0)
    global_std = np.std(np.vstack((real_data, synth_data)), axis=0)

    # Normalize the trajectories using global mean and standard deviation
    real_data = (real_data - global_mean) / global_std
    synth_data = (synth_data - global_mean) / global_std
    # Lists to store the FID scores
    fid_scores = []

    # List to store averages for code testing
    real_means = []
    synth_means = []

    # For each datapoint, obtain the mean a covariance for real and synth trajectories. Store the values in the lists
    for datapoint in range(flight_size):
        real_dtp = real_data[datapoint::flight_size]
        synth_dtp = synth_data[datapoint::flight_size]
    
        fid_dtp, real_mean, synth_mean = FID_dtp(real_dtp, synth_dtp)
        fid_scores = np.append(fid_scores, fid_dtp)
        real_means = np.append(real_means, real_mean[2]) ## number 2 correspond to altitude
        synth_means = np.append(synth_means, synth_mean[2])
    return fid_scores, real_means, synth_means ##the means are altitude means

def FID_dtp(real_dtp, synth_dtp):
    real_means_dtp = []
    synth_means_dtp = []

    ## Calculate mean of all parameters
    for i in range(len(real_dtp[0])):
        real_means_dtp = np.append(real_means_dtp, np.mean(real_dtp[:, i]))
        synth_means_dtp = np.append(synth_means_dtp, np.mean(synth_dtp[:, i]))

    ## Calculate covariance
    real_cov = np.cov(real_dtp, rowvar=False)
    synth_cov = np.cov(synth_dtp, rowvar=False)

    #Calculate FID
    mean_diff = real_means_dtp - synth_means_dtp
    cov_mean = sqrtm(np.dot(real_cov, synth_cov))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    fid_score = np.sum(mean_diff**2) + np.trace(real_cov + synth_cov - 2 * cov_mean)

    return fid_score, real_means_dtp, synth_means_dtp

fid = FID(r_data,s_data)

print("FID mean:", np.mean(fid[0])) ## mean FID for single valued comparison
# Plot FID scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fid[0])
plt.title('FID Scores - ' + str(route))
plt.xlabel('Datapoint')
plt.ylabel('FID Score')

# Plot real and synthetic trajectory altitud means
plt.subplot(1, 2, 2)
plt.plot(fid[1], label='Real Means')
plt.plot(fid[2], label='Synthetic Means')
plt.title('Real and Synthetic Trajectory Means')
plt.xlabel('Datapoint')
plt.ylabel('Mean Value')
plt.legend()

plt.tight_layout()
plt.show()