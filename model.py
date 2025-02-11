import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load dataset
df = pd.read_csv('news_dataset.csv')
df = df.dropna(subset=['text', 'label'])

# Features and Labels
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)

# Model Training
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)

print("Model and Vectorizer saved successfully!")
