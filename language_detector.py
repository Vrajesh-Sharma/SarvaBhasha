import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the data
data = pd.read_csv("Languages.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Ensure all data in the 'Language' column is in string format
data['language'] = data['language'].astype(str)

# Convert the text data and labels to numpy arrays
x = np.array(data["Text"])
y = np.array(data["language"])

# Initialize the CountVectorizer
cv = CountVectorizer()

# Transform the text data into feature vectors
X = cv.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and the CountVectorizer for later use
joblib.dump(model, 'language_detector_model.pkl')
joblib.dump(cv, 'count_vectorizer.pkl')