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

    return f'Latitude: {x}', f'Longitude: {y}', f'altitude: {z}', f'timedelta: {t}', f'flight_id: {f}', f'timestamp : {ts}'

#print(load_data(r"C:\Users\chris\OneDrive\Documenten\Chris\Q3 2024-2025\Project\EHAM_LIMC.csv"))


def convert_timestamp(files):
    from datetime import datetime
    import pandas as pd
    
    #Accessing data
    data = pd.read_csv(files)
    ts = data["timestamp"].to_numpy()


    stamp_list = []    
    for i in ts[:10]:
        time = datetime.fromisoformat(i)
        stamp = time.timestamp()
        stamp_list.append(stamp)
  
    return stamp_list

print(convert_timestamp(r"C:\Users\chris\OneDrive\Documenten\Chris\Q3 2024-2025\Project\EHAM_LIMC.csv"))
print(convert_timestamp(r"C:\Users\Gebruiker\Downloads\ESSA_LFPG.csv"))
