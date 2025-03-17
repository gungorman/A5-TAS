import numpy as np

#Function to load the Flight trajectory data



def load_data(files):
    import pandas as pd
    import numpy as np

    data = pd.read_csv(files)
    #data.drop(columns=["callsign", "icao24", "cluster", "timestamp"], inplace=True)
    x = data['latitude'].to_numpy()
    y = data['longitude'].to_numpy()
    z = data['altitude'].to_numpy()
    t = data['timedelta'].to_numpy()
    f = data['flight_id'].to_numpy()
    ts = data["timestamp"].to_numpy()

    #return f'Latitude: {x}', f'Longitude: {y}', f'altitude: {z}', f'timedelta: {t}', f'flight_id: {f}'
    return ts
    

#EHAM_LIMC
print(load_data(r"C:\Users\gungo\Downloads\ESSA_LFPG.csv"))

"""
def convert_timestamp(files):

    import pandas as pd
    import numpy as np

    data = pd.read_csv(files)
    ts = data["timestamp"].to_numpy()

    timestamp = 
"""

