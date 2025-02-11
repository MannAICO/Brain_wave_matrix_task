import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv('news_dataset.csv')
df = df.dropna(subset=['text', 'label'])

# Features and Labels
X = df['text']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization (Convert text to numerical representation)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'max_iter': [50, 100, 200]
}
pac = PassiveAggressiveClassifier()
grid_search = GridSearchCV(pac, param_grid, cv=5)
grid_search.fit(tfidf_train, y_train)

# Get the best parameters and re-train the classifier
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
best_pac = grid_search.best_estimator_

# Evaluate the model
tfidf_test = vectorizer.transform(X_test)
y_pred = best_pac.predict(tfidf_test)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_pac, model_file)

with open('vectorizer.pkl', 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vector_file:
    loaded_vectorizer = pickle.load(vector_file)

# Function to check if news is real or fake
def predict_news(news_text):
    # Transform the text using loaded vectorizer
    news_vector = loaded_vectorizer.transform([news_text])
    
    # Make prediction
    prediction = loaded_model.predict(news_vector)[0]
    
    return prediction


