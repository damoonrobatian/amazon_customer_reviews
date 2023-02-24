import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Text Classification

text classification is nothing but the process in which the text is assigned to particaular tagg or category depending upon its content. These classification can be used in real world problem for e.g Sentimental Analysis, Spam detection, Analyzing the Customer reviews and many more.

text classification classifiers can be used in organizing, structuring and categorizing for much as any type of text. The text from documents, medical studies and files and also all over the web. For this we are going to use Naive bayes classifier which is considered to be good for text classification.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score




#%% Load data
data_file = "/home/damoon/Dropbox/teaching/mcgill_teaching/data/amazon_customer_reviews/reviews_may19.csv"
data = pd.read_csv(data_file)
data.shape
data.head()
