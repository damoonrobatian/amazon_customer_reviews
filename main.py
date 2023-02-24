import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
#%%
'''
Text Classification
-------------------
Text classification is the process of learning to assign a label to a chunk of textual data through a model trained on a finite set of example data. The set of class labels is considered to be finite and the learning process is supervised. In other words, text classification is the application of classification to text data.  
Example applications include sentiment analysis, Spam detection, analysis of customer reviews etc. The text used for training and testing a model could be collected from a vast variety of sources such as formal documents, clinical studies, electronic correspondence, physician prescriptions, or online web pages. Any classification method might potentially be employed for this purpose. Here, we will compare some frequently used classification algorithms applied to our dataset.
'''





#%% Load data
data_file = "/home/damoon/Dropbox/teaching/mcgill_teaching/data/amazon_customer_reviews/reviews_may19.csv"
data = pd.read_csv(data_file)
data.shape
data.head()


#%% Replace the text with numbers
df['new_Labels'] =   df['Labels'].apply(lambda v: 1 if v=='Positive' else 0)

'''
Here we have created a new column as "new_Labels" which contains the integer values of "Labels" column, for "Positive" we have replaced it with "1" and for "Negative" we have replaced it with "0".
'''
df.head()
df.tail()

#%% Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['Customer_Reviews'], df['new_Labels'], random_state=1)
vectorizer = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = vectorizer.fit_transform(X_train)
X_test_cv = vectorizer.transform(X_test)

#%% Convert the Customer_Reviews into word count vectors
Word_frequency = pd.DataFrame(X_train_cv.toarray(), columns=vectorizer.get_feature_names())
top_words = pd.DataFrame(Word_frequency.sum()).sort_values(0, ascending=False)
print(Word_frequency, '\n')
print(top_words)

'''
Here in the above we have converted the Reviews into vectors, As the naive bayes classifier needs to be able to calculate how many times each word appears in each document and how many times it appears in each category. for Conversion we have used count vectorizer, and also you can see the word frequency and top words in the above.
'''
#%% Fit the model and make the predictions
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

#%% Print the results
print('Accuracy score for Customer Reviews model is: ', accuracy_score(y_test, predictions), '\n')
print('Precision scorefor Customer Reviews model is: ', precision_score(y_test, predictions), '\n')

'''
As these are the results based on a sample dataset that only have 10 records, but for more data it will give us more better results. Now we will understand what accuracy and precision score tell us:
Accuracy Score will tell us that out of all the identifications that we have made how many are correct.
Precision Score will tell us that out of all the positive/negative identification we made how many are correct.
'''
