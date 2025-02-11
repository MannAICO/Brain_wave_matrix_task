from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        return render_template('index.html', prediction_text=f'The news is: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
