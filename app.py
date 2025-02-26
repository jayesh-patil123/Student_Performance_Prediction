from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
app = Flask(__name__)

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect input data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),  # ✅ Fixed field mapping
            writing_score=float(request.form.get('writing_score'))   # ✅ Fixed field mapping
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        # Run Prediction Pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])

# Application Entry Point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Dynamic port binding for Render

    if os.name == "nt":  # ✅ Windows (use Waitress)
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
    else:  # ✅ Render/Linux (Gunicorn will be used externally)
        app.run(host="0.0.0.0", port=port)
