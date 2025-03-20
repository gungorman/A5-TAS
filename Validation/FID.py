## FID is a method that calculates the distance between distribution of datapoints.
## Since out data is high dimensional, for each time step, one feature of each trajectory will be taking
## This will result in distributions of real and synthetic data, one for each datapoint and feature
## Finally, the FID is calculated for each of these distributions, obtaining a FID score in time for each feature

import numpy as np
import matplotlib.pyplot as plt

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

    # Lists to store values of average and covariance
    real_mean = [] ## List for storing av in real data features
    real_cov = [] ## List for storing covariance of real data features
    synth_mean = [] ## Same for synth
    synth_mean = [] ## Same for synth

    # For each datapoint, obtain the mean a covariance for real and synth trajectories. Store the values in the lists
    for datapoint in range(flight_size):
        real_dtp = real_data[datapoint::flight_size]
        synth_dtp = synth_data[datapoint::flight_size]
        real_means_dtp = []
        covariance = []
        for i in range(len(real_dtp[0])):
            real_means_dtp = np.append(real_means_dtp, np.mean(real_dtp[:, i]))
        real_mean = np.append(real_mean, real_means_dtp)
    print(real_mean) ## To do, fix means to be in separate rows in the general array
FID(data, data)