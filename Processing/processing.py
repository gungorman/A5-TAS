import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_trajectories_comparison(array1, array2, 
                                 titles=('Trajectories 1', 'Trajectories 2'),
                                 figsize=(16, 6), linewidth=1.5, alpha=0.7):
    """
    Plot two sets of flight trajectories side-by-side with altitude-based color gradients.
    Color bar is placed on the right side of the figure, outside the subplots.
    
    Parameters:
        array1 (np.ndarray): First trajectory array (num_traj, timesteps, features)
        array2 (np.ndarray): Second trajectory array (num_traj, timesteps, features)
        titles (tuple): Titles for each subplot
        figsize (tuple): Overall figure size (width, height)
        linewidth (float): Width of trajectory lines
        alpha (float): Transparency of lines
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Create a colormap for altitude (using viridis)
    cmap = plt.get_cmap('viridis')
    
    # Determine global min/max for consistent coloring
    global_alt_min = min(array1[:, :, 2].min(), array2[:, :, 2].min())
    global_alt_max = max(array1[:, :, 2].max(), array2[:, :, 2].max())
    norm = plt.Normalize(global_alt_min, global_alt_max)
    
    def plot_single_array(ax, array, title):
        """Helper function to plot one array"""
        for traj in array:
            lats = traj[:, 0]  # Latitude
            lons = traj[:, 1]  # Longitude
            alts = traj[:, 2]  # Altitude
            
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
            lc.set_array(alts)
            ax.add_collection(lc)
        
        ax.set_xlim(min(array1[:, :, 1].min(), array2[:, :, 1].min()),
                   max(array1[:, :, 1].max(), array2[:, :, 1].max()))
        ax.set_ylim(min(array1[:, :, 0].min(), array2[:, :, 0].min()),
                   max(array1[:, :, 0].max(), array2[:, :, 0].max()))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.grid(True)
    
    # Plot both arrays
    plot_single_array(ax1, array1, titles[0])
    plot_single_array(ax2, array2, titles[1])
    
    # Create an axis for the colorbar on the right side
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label='Altitude')
    
    plt.tight_layout()
    plt.show()


def load_single_array(npz_file_path, array_name):
    """
    Load a single NumPy array from an .npz file.
    
    Parameters:
        npz_file_path (str): Path to the .npz file
        array_name (str): Name of the array to load
        
    Returns:
        np.ndarray: The requested NumPy array
        
    Raises:
        FileNotFoundError: If the .npz file doesn't exist
        KeyError: If the specified array doesn't exist in the file
    """
    try:
        with np.load(npz_file_path) as data:
            if array_name not in data:
                available = list(data.keys())
                raise KeyError(
                    f"Array '{array_name}' not found. "
                    f"Available arrays: {available}"
                )
            return data[array_name]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {npz_file_path}")
    
result1 = load_single_array(r"C:\Users\gungo\Downloads\EHAM_LIMC_train_dataa_n=40.npz", "data")
print(result1.shape)
result2 = load_single_array(r"C:\Users\gungo\OneDrive\Desktop\A05 Data\EHAM_LIMC_train_filtered_data_final_n=40.npz", "data")
print(result2.shape)
array1 = result1
array2 = result2


# Plot comparison
plot_trajectories_comparison(
    array1, 
    array2,
    titles=('Unfiltered Trajectories', 'Filtered Trajectories'),
    figsize=(18, 7)
)