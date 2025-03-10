import numpy as np

def load_data(files):
    data = np.load(files)
    
    extracted_data = {name: data[name] for name in data.files}
    
    return extracted_data


print(load_data(r"C:\\Users\\chris\\OneDrive\\Documenten\\Chris\\Q3 2024-2025\\Project\\stockv_subsampled_train_perc_10.npz"))
