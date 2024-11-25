import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Load dataset
raw_mail_data = pd.read_csv("D:/created websites/Spam Mail Prediction/mail_data.csv")

# Replace null values with an empty string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate the data as text and label
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Save the model and the vectorizer
dump(model, 'logistic_regression_model.joblib')
dump(feature_extraction, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully.")
