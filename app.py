from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained spam detection model (make sure it's in the same folder)
with open('spam.pkl', 'rb') as file:
    model = pickle.load(file)

# Load your vectorizer (used for text transformation)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['Message']

        # Transform text using the same vectorizer used in training
        transformed_text = vectorizer.transform([email_text])
        
        # Predict using the loaded model
        prediction = model.predict(transformed_text)[0]

        result = "Spam Email ☠️" if prediction == 0 else "Not spam Email 🙂"
        return render_template('index.html', pred=result)

if __name__ == "__main__":
    app.run(debug=True)
