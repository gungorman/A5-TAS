import numpy as np
import matplotlib.pyplot as plt

#import NPZ file here for generated data

## Import the data EDIT THE ROUTE
route = "EHAM_LIMC"
n = 20

#TimeVAE/outputs/gen_data/EHAM_LIMC_train_dataa_n=20
#TimeVAE\data\EHAM_LIMC_train_dataa_n=20.npz


real_data_npz = np.load(r"C:\Users\solif\Desktop\folderfolder\ESSA_LFPG_val_data_n=40.npz")
fake_data_npz = np.load(r"C:\Users\solif\Desktop\folderfolder\ESSA_LFPG_gen_data_n=40_L=50.npz")

real_data = real_data_npz['val']
fake_data = fake_data_npz['data']

r_data = [item for sublist in real_data for item in sublist]
s_data = [item for sublist in fake_data for item in sublist]

r_data = np.array(r_data)
s_data = np.array(s_data)




def autocorrelation(series, max_lag):
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    acf = np.array([
        np.correlate(series[:n-lag] - mean, series[lag:] - mean, mode='valid')[0] / (var * (n - lag))
        for lag in range(1, max_lag + 1)
    ])
    return acf

def compute_acd(real, fake, max_lag):
    acd_x = np.abs(autocorrelation(real[:, 0], max_lag) - autocorrelation(fake[:, 0], max_lag))
    acd_y = np.abs(autocorrelation(real[:, 1], max_lag) - autocorrelation(fake[:, 1], max_lag))
    acd_z = np.abs(autocorrelation(real[:, 2], max_lag) - autocorrelation(fake[:, 2], max_lag))
    
    return np.mean([acd_x, acd_y, acd_z], axis=0)  # Aggregate over dimensions

def compute_speed_magnitude(data):
    diff = np.diff(data, axis=0)  # Compute differences between consecutive points
    speed = np.linalg.norm(diff, axis=1)  # Compute Euclidean speed
    return speed


# Simulate real and fake flight paths (latitude, longitude, altitude)
#n_points = number of data points
#add generated data as an array

max_lag = 200

# Compute ACD
acd_values = compute_acd(r_data, s_data, max_lag)
print(f"Mean ACD over {max_lag} lags:", np.mean(acd_values))

# Compute speed autocorrelation
real_speed = compute_speed_magnitude(r_data)
fake_speed = compute_speed_magnitude(s_data)

real_speed_acf = autocorrelation(real_speed, max_lag)
fake_speed_acf = autocorrelation(fake_speed, max_lag)

acd_speed = np.abs(real_speed_acf - fake_speed_acf)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(acd_values, label="ACD (Latitude, Longitude, Altitude)", color="blue")
plt.plot(acd_speed, label="ACD (Speed)", color="red", linestyle="dashed")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation Difference")
plt.legend()
plt.title("Autocorrelation Difference for Flight Path")
plt.show()
