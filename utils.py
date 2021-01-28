import numpy as np
from geopy.distance import geodesic as geodesic
from pandas.io.parsers import count_empty_vals
import torch
from matplotlib import pyplot as plt

def extract_timestamp(timestamp):
    
  time = timestamp.split()[1]
  hour = time[:2]
  mins = time[3:5]

  return int(hour), int(mins)

def add_hours_mins_cols(data):
  timestamps = data["TIMESTAMP"]
  hours_arr = np.array(list(map(lambda x: extract_timestamp(x)[0], timestamps)))
  mins_arr = np.array(list(map(lambda x: extract_timestamp(x)[1], timestamps)))

  data.insert(0, "TIMESTAMP_HOURS", hours_arr)
  data.insert(0, "TIMESTAMP_MINUTES", mins_arr)

def get_distance_from_coords(lat1, long1, lat2, long2):
    
    return geodesic((lat1, long1), (lat2, long2)).km

def extract_coords(data):
    
    user_lats = data["USER_LAT"].values
    user_longs = data["USER_LONG"].values
    venue_lats = data["VENUE_LAT"].values
    venue_longs = data["VENUE_LONG"].values

    coords = np.stack((user_lats, user_longs, venue_lats, venue_longs), axis=1)
    print(coords.shape)
    return coords

def add_distance_col(data):
    
    coords = extract_coords(data)
    distances_arr = np.array(
        list(map(
            lambda x: get_distance_from_coords(
                x[0],
                x[1],
                x[2],
                x[3]), coords))
    )

    print(distances_arr[5:10])
    data.insert(0, "DISTANCES_KM", distances_arr)

def split_data(X, train_split_percentage):
    
    split_idx = int(X.shape[0] // (1/train_split_percentage))

    chunk1 = X[:split_idx]
    chunk2 = X[split_idx:]

    return chunk1, chunk2

def extract_timestamp(timestamp):
    
  time = timestamp.split()[1]
  hour = time[:2]
  mins = time[3:5]

  return int(hour), int(mins)

def torch_np(np_array, dtype=np.float32):
    return torch.from_numpy(np_array.astype(dtype))

def scatter_all_parameters(y, x):
    
    # plot actual delivery against:
    # - item count 
    # - distance_km 
    # - timestamp hours 
    # - timestampt mins 
    # - wind speed 
    # - precipitation 
    # - cloud coverage 
    # - temperature 
    # - estimated delivery in minutes

    col_names = ["ITEM_COUNT", "DISTANCES_KM", "TIMESTAMP_HOURS", "TIMESTAMP_MINUTES",
                "WIND_SPEED", "PRECIPITATION", "CLOUD_COVERAGE", "TEMPERATURE", 
                "ESTIMATED_DELIVERY_MINUTES"]

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
    for i, ax in enumerate(axs.flat):
        
        scatter_x = x[col_names[i]]
        scatter_y = y
        ax.scatter(scatter_x, scatter_y)
        ax.set_title(col_names[i])