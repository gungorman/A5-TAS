# Import required packages
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA  
import matplotlib.pyplot as plt 
import numpy as np  


# Define the data diversity assessment function
def data_diversity(ori_data, generated_data, analysis, average_dimension='sequence', max_sample_size=1000):
    """
    Data diversity assessment using PCA or t-SNE for original & synthetic distribution visualization.


    Inputs:
        - ori_data (array): original data
        - synthetic_data (array): synthetic data
        - analysis_type (str): PCA or t-SNE
        - average_dimension (string): flatten along 'sequence' or 'samples' dimension
        - max_sample_size (int): maximum sample size for computational speed
    """
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    
    # Determine the analysis sample size (minimum of 1000 or the length of the original data)
    anal_sample_no = min(max_sample_size, len(ori_data))


    # Randomly permute indices for data preprocessing
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]


    # Convert original and generated data to numpy arrays and select a subset based on indices
    ori_data, generated_data = np.asarray(ori_data)[idx], np.asarray(generated_data)[idx]


    if average_dimension =='sequence':
        # Compute the mean along the sequence length dimension for both datasets
        prep_data = np.mean(ori_data[:, :, :], axis=1)
        prep_data_hat = np.mean(generated_data[:, :, :], axis=1)
    
    elif average_dimension =='samples':
        prep_data = np.mean(ori_data[:, :, :], axis=2)
        prep_data_hat = np.mean(generated_data[:, :, :], axis=2)
        
    else:
        prep_data = ori_data.reshape(ori_data.shape[0], -1)
        prep_data_hat = generated_data.reshape(generated_data.shape[0], -1)
            
    # Define colors for visualization (red for original, blue for synthetic)
    colors = ["red"] * anal_sample_no + ["blue"] * anal_sample_no


    # Perform analysis based on user choice (PCA or t-SNE)
    if analysis == 'PCA':
        # Apply PCA to both original and synthetic data
        pca_results = PCA(n_components=2).fit_transform(prep_data)
        pca_hat_results = PCA(n_components=2).fit_transform(prep_data_hat)


        # Plot PCA results
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:anal_sample_no], alpha=0.35, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[anal_sample_no:], alpha=0.35, label="Synthetic")


    elif analysis == 't-SNE':
        # Combine preprocessed data for t-SNE analysis
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)


        # Apply t-SNE to combined data
        tsne_results = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(prep_data_final)


        # Plot t-SNE results
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], c=colors[:anal_sample_no], alpha=0.35, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], c=colors[anal_sample_no:], alpha=0.35, label="Synthetic")


    # Add legend and labels to the plot
    # plt.legend()
    plt.title('PCA Plot' if analysis == 'PCA' else 't-SNE Plot')
    plt.xlabel('x-pca' if analysis == 'PCA' else 'x-tsne')
    plt.ylabel('y-pca' if analysis == 'PCA' else 'y-tsne')
    plt.savefig('diversity', dpi=400, bbox_inches='tight')


    # Display the plot
    plt.show()






if __name__ == '__main__':
    
    # Example usage
    ori_data = np.random.rand(100, 10, 5)  # 100 samples, each with a sequence of length 10 and 5 features
    generated_data = np.random.rand(100, 10, 5)  # Synthetic data with the same shape as ori_data
    data_diversity(ori_data, generated_data, 'PCA', 'sequence')
    data_diversity(ori_data, generated_data, 't-SNE')
    
    # Single cluster
    trajectories = np.load('S175R14T180clust5.npz')['X']
    labels = np.load('S100R14T180clust5.npz')['y']
    unique_labels = np.unique(labels)
    separated_arrays = {}
    for label in unique_labels:
        separated_arrays[label] = trajectories[labels == label]


    cluster0 = separated_arrays[0]
    cluster1 = separated_arrays[1]
    cluster2 = separated_arrays[2]
    cluster3 = separated_arrays[3]
    cluster4 = separated_arrays[4]
    
    clust0 = np.load('clust0.npy')
    clust1 = np.load('clust1.npy')
    clust2 = np.load('clust2.npy')
    clust3 = np.load('clust3.npy')
    clust4 = np.load('clust4.npy')
    
    original = np.concatenate((cluster0, cluster1, cluster2, cluster3, cluster4))
    generated = np.concatenate((clust0, clust1, clust2, clust3, clust4))
