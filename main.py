import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% Load data
data_file = "/home/damoon/Dropbox/teaching/mcgill_teaching/data/amazon_customer_reviews/reviews_may19.csv"
data = pd.read_csv(data_file)
data.shape
data.head()
