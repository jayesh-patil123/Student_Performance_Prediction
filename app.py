from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            # Log received form data
            print("📥 Received Form Data:", request.form)

            # Extract form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),  # Fixed swapped fields
                writing_score=float(request.form.get('writing_score'))   # Fixed swapped fields
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("📝 DataFrame Created:", pred_df)

            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("🔮 Prediction:", results)

            return render_template('home.html', results=results[0])

        except Exception as e:
            print("❌ Error:", str(e))
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
