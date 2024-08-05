from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

model = joblib.load('random_forest_model.joblib')


@app.route('/')
def home():
    return """Welcome to the prediction API!
              Use the /predict endpoint to get predictions."""


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    response = {
        'prediction': int(prediction[0])
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
