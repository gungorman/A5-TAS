## FID is a method that calculates the distance between distribution of datapoints.
## Since out data is high dimensional, for each time step, one feature of each trajectory will be taking
## This will result in distributions of real and synthetic data, one for each datapoint and feature
## Finally, the FID is calculated for each of these distributions, obtaining a FID score in time for each feature

import numpy as np

## Import the data and take just the first 4 columns, removes the title column
data = np.genfromtxt('C:/Sofia/TU Delft/Bsc2/Proyect/second semester/EHAM_LIMC.csv', delimiter=',')

def FID(real_data, synth_data):
    ## Clean the data to obtain an array with just the 4 relevant info
    real_data = np.delete(real_data, 0, 0)
    real_data = np.delete(real_data, np.s_[4:], 1)
    synth_data = np.delete(synth_data, 0, 0)
    synth_data = np.delete(synth_data, np.s_[4:], 1)

    ## look for starts
    real_index = []
    synth_index = []
    for i in range(len(real_data)):
        if real_data[i][3] == 0:
            real_index.append(i)
#    for i in range(len(synth_data)):
#        if synth_data[i][3] == 0:
#            synth_index.append(i)
    flight_size = real_index[1]
    alt_real_mean = [] ## List for storing av altitude in real data
    alt_real_cov = [] ## List for storing covariance of altitude of real data
    alt_real = real_data[i:, 2] ## List with all real altitudes
    print(alt_real)
    for i in range(flight_size):
        alt_dt = alt_real[i::flight_size]
FID(data, data)