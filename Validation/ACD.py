import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(series, max_lag):
    """Compute autocorrelation for a given time series up to max_lag."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    acf = np.array([
        np.correlate(series[:n-lag] - mean, series[lag:] - mean, mode='valid')[0] / (var * (n - lag))
        for lag in range(1, max_lag + 1)
    ])
    return acf

def compute_acd(real, fake, max_lag):
    """Compute Autocorrelation Difference (ACD) for real vs. synthetic data."""
    acd_x = np.abs(autocorrelation(real[:, 0], max_lag) - autocorrelation(fake[:, 0], max_lag))
    acd_y = np.abs(autocorrelation(real[:, 1], max_lag) - autocorrelation(fake[:, 1], max_lag))
    acd_z = np.abs(autocorrelation(real[:, 2], max_lag) - autocorrelation(fake[:, 2], max_lag))
    
    return np.mean([acd_x, acd_y, acd_z], axis=0)  # Aggregate over dimensions

def compute_speed_magnitude(data):
    """Compute speed magnitude at each time step."""
    diff = np.diff(data, axis=0)  # Compute differences between consecutive points
    speed = np.linalg.norm(diff, axis=1)  # Compute Euclidean speed
    return speed

# Example Usage
np.random.seed(42)

# Simulate real and fake flight paths (latitude, longitude, altitude)
n_points = 1000
real_flight = np.cumsum(np.random.randn(n_points, 3), axis=0)  # Random walk
fake_flight = np.cumsum(np.random.randn(n_points, 3), axis=0) * 1.1  # Slightly different

max_lag = 200

# Compute ACD
acd_values = compute_acd(real_flight, fake_flight, max_lag)
print(f"Mean ACD over {max_lag} lags:", np.mean(acd_values))

# Compute speed autocorrelation
real_speed = compute_speed_magnitude(real_flight)
fake_speed = compute_speed_magnitude(fake_flight)

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
