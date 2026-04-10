from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
from preprocessor import StormDamagePreprocessor

app = Flask(__name__)

model = xgb.XGBRegressor()
model.load_model('../models/storm_damage_model.json')
preprocessor = StormDamagePreprocessor(models_dir='../models')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Event naarrative is what we use sentence transformers / embedding features on
        if not data.get('EVENT_NARRATIVE') or len(data['EVENT_NARRATIVE']) < 10:
            return jsonify({'error': 'EVENT_NARRATIVE required (min 10 chars)'}), 400
        
        features = preprocessor.preprocess(data)
        log_prediction = model.predict(features)[0]
        
        # Model was trained on log transformed damage, so we must invert here
        prediction = np.expm1(log_prediction)
        
        # Cap at 500,000 to control outliers
        prediction = min(prediction, 500000)
        
        response = {
            'predicted_damage_usd': float(prediction),
            'model_r_squared': 0.70
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
