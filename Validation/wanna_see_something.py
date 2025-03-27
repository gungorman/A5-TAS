import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import glob
import matplotlib.cm as cm  # For colormap

# === CONFIG ===
data_folder = 'TimeVAE\\outputs\\generated_data'  # Use double backslashes
file_pattern = '*.npz'

# === READ & MERGE FILES ===
npz_files = glob.glob(os.path.join(data_folder, file_pattern))
flight_data = []

for file in npz_files:
    try:
        data = np.load(file)
        # The data array should have the shape (n_samples, n_timesteps, 4)
        trajectory = data['data']  # Assuming the array of interest is stored in 'data'
        
        # Extract latitude, longitude, and altitude from the array
        latitude = trajectory[:, :, 0]  # Latitude is the first column
        longitude = trajectory[:, :, 1]  # Longitude is the second column
        altitude = trajectory[:, :, 2]  # Altitude is the third column (or other metric)

        # Store data
        flight_data.append({
            'filename': os.path.basename(file),
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        })
    except Exception as e:
        print(f"Failed to load {file}: {e}")

if not flight_data:
    print("No valid .npz files with latitude and longitude found.")
    exit()

# Number of flight paths to display
n_paths = 5  # Change this value as needed

# === OPTIONAL: PLOT 3D FLIGHT PATHS ===
has_altitude = any(flight['altitude'] is not None for flight in flight_data)

if has_altitude:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap
    colormap = cm.viridis  # You can choose other colormaps like 'plasma', 'inferno', etc.
    colors = colormap(np.linspace(0, 1, n_paths))  # Generate `n_paths` distinct colors from the colormap

    for i, flight in enumerate(flight_data[:n_paths]):  # Only plot the first `n_paths` entries
        ax.plot(flight['longitude'].flatten(), flight['latitude'].flatten(), flight['altitude'].flatten(),
                label=flight['filename'], color=colors[i])  # Assign a unique color to each flight path

    ax.set_title('3D Flight Paths')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (m)')
    ax.legend()
    plt.show()
