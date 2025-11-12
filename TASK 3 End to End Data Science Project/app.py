from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return "Titanic Survival Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'survived': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
