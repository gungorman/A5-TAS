import pandas as pd
import requests
from io import StringIO
from tqdm import tqdm

# URL of the raw CSV file on Zenodo
url = "https://zenodo.org/record/13767132/files/EHAM_LIMC.csv"  # Replace with the actual file URL

# Send a GET request to download the file, stream=True to allow progress tracking
response = requests.get(url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Get the total size of the file
    total_size = int(response.headers.get('content-length', 0))
    
    # Create a progress bar using tqdm
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading CSV") as pbar:
        # Read the file in chunks and update the progress bar
        content = []
        for chunk in response.iter_content(chunk_size=1024):
            content.append(chunk)
            pbar.update(len(chunk))
        
    # Join the chunks together and load into pandas
    content = b''.join(content)
    data = pd.read_csv(StringIO(content.decode('utf-8')))
    
    print(data)  # Show the first few rows of the data
else:
    print(f"Failed to download the file. Status code: {response.status_code}")
