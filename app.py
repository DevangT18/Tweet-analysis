# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)    

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict_sentiment():
    tweet = request.form['tweet']

    # Vectorize input tweet
    vectorized_tweet = vectorizer.transform([tweet])

    # Make prediction
    prediction = model.predict(vectorized_tweet)

    # Convert prediction to readable format
    output = 'Positive' if prediction[0] == 1 else 'Negative'

    return render_template('index.html', prediction_text=f'Sentiment: {output}')

if __name__ == "__main__":
    app.run(debug=True)