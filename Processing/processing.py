import numpy as np

def load_data(files):
    data = np.load(files)
    
    extracted_data = {name: data[name] for name in data.files}
    
    return extracted_data


print(load_data(r"Processing\air_subsampled_train_perc_10.npz"))
