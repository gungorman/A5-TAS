import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import requests
import io

url = "https://zenodo.org/record/13767132/files/EHAM_LIMC.csv"

print("Downloading file...")
with requests.get(url, stream=True) as response:
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    file_content = io.BytesIO()
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file_content.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete")

file_content.seek(0)
df = pd.read_csv(file_content, names=['latitude','longitude','altitude','timedelta','flight_id', 'callsign', 'icao24', 'cluster', 'timestamp'], skiprows= 1)

print("Performing PCA...")

features = ['latitude', 'longitude', 'altitude', 'timedelta']
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
with tqdm(total=1, desc="PCA Progress") as pbar:
    principalComponents = pca.fit_transform(x)
    pbar.update(1)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['timedelta']]], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)


timedelta_values = np.linspace(-1, 1, num=3)  # Define three values for colors
colors = ['r', 'g', 'b']

for t_val, color in zip(timedelta_values, colors):
    indicesToKeep = finalDf['timedelta'] == t_val
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color, s=50)

ax.legend([f"Timedelta {t}" for t in timedelta_values])
ax.grid()
plt.show()
