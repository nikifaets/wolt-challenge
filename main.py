import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np 
from matplotlib import pyplot as plt
from utils import *


dataset_path = "data-science-summer-intern-2021/orders_autumn_2020.csv"
data = pd.read_csv(dataset_path).dropna()

add_hours_mins_cols(data)
add_distance_col(data)

print(data.columns)
X_names = ["ITEM_COUNT",
        "DISTANCES_KM",
        "TIMESTAMP_HOURS",
        "TIMESTAMP_MINUTES",
        "ESTIMATED_DELIVERY_MINUTES"]

Y_names = ["ACTUAL_DELIVERY_MINUTES"]

X = data[X_names]
Y = data[Y_names]

scatter_all_parameters(Y, data)



