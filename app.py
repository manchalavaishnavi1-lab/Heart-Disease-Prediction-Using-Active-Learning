# app.py

from flask import Flask, render_template, request
from model import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():

    data = [
        float(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        float(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        int(request.form['slope']),
        int(request.form['ca']),
        int(request.form['thal'])
    ]

    pred, prob = predict(data)

    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    return render_template('index.html', prediction=result, probability=round(prob, 3))

if __name__ == "__main__":
    app.run(debug=True)