## FID is a method that calculates the distance between distribution of datapoints.
## Since out data is high dimensional, for each time step, one feature of each trajectory will be taking
## This will result in distributions of real and synthetic data, one for each datapoint and feature
## Finally, the FID is calculated for each of these distributions, obtaining a FID score in time for each feature

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

## Import the data and take just the first 4 columns, removes the title column
data = np.genfromtxt('C:/Sofia/TU Delft/Bsc2/Proyect/second semester/EHAM_LIMC.csv', delimiter=',')
def FID(real_data, synth_data):
    ## Clean the data to obtain an array with just the 4 relevant info
    real_data = np.delete(real_data, 0, 0)
    real_data = np.delete(real_data, np.s_[4:], 1)
    synth_data = np.delete(synth_data, 0, 0)
    synth_data = np.delete(synth_data, np.s_[4:], 1)

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

    # Lists to store the FID scores
    fid_scores = []

    # For each datapoint, obtain the mean a covariance for real and synth trajectories. Store the values in the lists
    for datapoint in range(flight_size):
        real_dtp = real_data[datapoint::flight_size]
        synth_dtp = synth_data[datapoint::flight_size]
    
        fid_dtp = FID_dtp(real_dtp, synth_dtp)
        fid_scores = np.append(fid_scores, fid_dtp)
    return fid_scores

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
    if fid_score < 0:
        pass

    return fid_score

fid = FID(data, data)
plt.plot(range(len(fid)), fid)
plt.show()