import tensorflow as tf
import numpy as np
import pandas as pd

data=pd.read_csv("data/spotify_dataset_train.csv")

print("shape:", np.shape(data))

data.info()
