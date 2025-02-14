Overview :- 

![ML_detection](https://github.com/user-attachments/assets/c6df0916-7d32-4fff-a5b0-379b61f6a166)


Explain the steps that i have taken :-

1. Importing Required Libraries:

pandas: For data manipulation and reading the CSV file.
Scikit-learn Modules:
train_test_split to split the data into training and testing sets.
GridSearchCV to perform hyperparameter tuning.
TfidfVectorizer to convert text into numerical features using TF-IDF.
PassiveAggressiveClassifier as the classification algorithm for distinguishing between real and fake news.
classification_report to evaluate model performance.
pickle: To save and load the trained model and vectorizer.
Flask: (Mentioned in your project) Used to create a web application that serves as the frontend, allowing users to input news text and receive predictions.

2.Loading and Cleaning the Dataset:

The dataset is loaded from a CSV file (news_dataset.csv).
Any rows with missing values in the text or label columns are removed to ensure data quality.
Preparing Features and Labels:

Features (X): The text column containing the news articles.
Labels (y): The label column that indicates whether the news is "real" or "fake."

3.Splitting the Data:

The data is split into training and testing sets (typically 80% for training and 20% for testing) to evaluate the model's performance on unseen data.
A fixed random state (e.g., 42) ensures reproducibility of the results.

4.Text Vectorization:

TfidfVectorizer is used to transform the textual data into a numerical format that the machine learning model can work with.
Parameters like stop_words='english' and max_df=0.7 help filter out common and overly frequent words, reducing noise in the data.

5.Hyperparameter Tuning with GridSearchCV:

A parameter grid is defined to test different values for hyperparameters (e.g., C and max_iter) of the Passive Aggressive Classifier.
GridSearchCV is used with 5-fold cross-validation to systematically evaluate combinations of parameters and choose the best configuration.

6.Model Training and Evaluation:

The best model (with the optimal hyperparameters) is selected from GridSearchCV.
The test set is vectorized using the same TF-IDF vectorizer, and the model makes predictions on these unseen data.
A classification report is generated, providing metrics like precision, recall, and f1-score to assess the model's performance.

7.Saving the Model and Vectorizer:

The trained model and the TF-IDF vectorizer are saved to disk using pickle.
This enables you to load them later for making predictions without needing to retrain the model.

8.Loading the Model and Vectorizer for Prediction:

The saved model and vectorizer are loaded from disk, making them ready for use in a production environment or for future predictions.

9.Prediction Function:

A function (predict_news) is defined to take in new news text, transform it using the loaded vectorizer, and then use the loaded model to predict whether the news is real or fake.
This function returns the predicted label.

10.Flask Integration for Frontend:

Flask is used to create a web application that serves as the frontend for your project.
The Flask app includes routes that allow users to enter news text into a web form.
Upon submission, the app calls the predict_news function, processes the input text, and displays the prediction (real or fake) back to the user.
This integration makes your project interactive and accessible through a web browser, bridging the gap between the machine learning backend and the user interface.
