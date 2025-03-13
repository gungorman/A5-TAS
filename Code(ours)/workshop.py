import pandas as pd
import numpy as np
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('Code(ours)/traj.csv')
# Convert the DataFrame to a Numpy array
array=df[['latitude','longitude','altitude','timedelta']].values
#array=array_n["latitude"]#,"longitude","altitude","timedelta"]]
# Save the array to a NPZ file
np.savez('Code(ours)/data.npz', array)



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