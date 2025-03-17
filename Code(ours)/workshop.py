import pandas as pd
import numpy as np
from datetime import datetime

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('Code(ours)/LOWW_EGLL - Copy.csv')
#unique_flight_ids = df['flight_id'].unique()
#print(unique_flight_ids)
# Convert the DataFrame to a Numpy array
array=df[['flight_id', 'timestamp', 'altitude']].values # 'latitude','longitude','altitude']].values
MainList = []
for i in range(len(array)):
    MainList.append([array[i][0], array[i][1], array[i][2]])
    
#int(round(array[i][1].timestamp()))
for i in range(len(array)):
    array[i][1]=int(round(datetime.fromisoformat(array[i][1]).timestamp()))
#datetime.strptime(array[i][1], '%Y-%m-%d %H:%M:%S')
# Save the array to a NPZ file
np.savez('Code(ours)/data.npz', array)
flight_id='TRAJ_1001'
flight_id_numeric = int(flight_id.split('_')[-1])
print(flight_id_numeric)
print(array.shape)

data=np.load('Code(ours)/data.npz', allow_pickle=True)
lst=data.files
for item in lst:
    print(item)
    print(data[item])

"""
#loading data:
import pandas as pd
data = pd.read_csv('filename.csv')
data.head()


print(data)
n=100000 #number of lines considered
i=1
number=1
while i<n:
    if round(data[3], 4)==0:
        number+=1
        print(f"new flight path found: {number} ")
    i+=1

print("finished running")"""