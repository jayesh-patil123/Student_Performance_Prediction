🎓 Student Performance Prediction App



🚀 Overview

The Student Performance Prediction App is an end-to-end machine learning project designed to predict student performance based on various input features. The project follows a structured ML pipeline, from data ingestion to model deployment, using Python, Flask, CatBoost, and Scikit-Learn.

📂 Project Structure

Student-Performance-Prediction/
│-- artifacts/               # Stores trained models, preprocessor, and other artifacts

│-- catboost_info/           # Logs and metadata from CatBoost model training

│-- data/                    # Raw dataset and train-test split

│   │-- data.csv             # Raw dataset

│   │-- train.csv            # Training dataset

│   │-- test.csv             # Testing dataset

│-- logs/                    # Log files for debugging and monitoring

│-- notebook/                # Jupyter Notebooks for EDA and testing

│-- src/                     # Source code directory

│   │-- components/          # Data processing and training scripts

│   │   │-- Data_ingestion.py

│   │   │-- Data_transformation.py

│   │   │-- Model_trainer.py

│   │-- pipeline/            # Pipeline for training and prediction

│   │   │-- Train_pipeline.py

│   │   │-- Predict_pipeline.py

│   │-- exception.py         # Custom exception handling

│   │-- logger.py            # Logging utilities

│   │-- utils.py             # Helper functions

│-- templates/               # HTML templates for the web interface

│   │-- home.html

│   │-- index.html

│-- app.py                   # Main Flask application

│-- requirements.txt         # Python dependencies

│-- README.md                # Project documentation

│-- Training.log             # Logs for model training

✨ Features

✅ End-to-End ML Pipeline (Data Ingestion ➝ Transformation ➝ Model Training ➝ Prediction)✅ CatBoost Model for Performance Prediction✅ Flask Web Interface for User Input & Predictions✅ Logging and Exception Handling for Debugging✅ Pretrained Model & Preprocessor Saved for Deployment✅ Supports Training New Models with Additional Data

🔧 Installation

1️⃣ Clone the Repository

git clone [https://github.com/jayesh-patil123/Student-Performance-Prediction.git](https://github.com/jayesh-patil123/Student_Performance_Prediction)

cd Student-Performance-Prediction

2️⃣ Create a Virtual Environment & Install Dependencies

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

🎯 Usage

1️⃣ Run the Flask App

python app.py

2️⃣ Open in Browser

Go to http://127.0.0.1:5000/ to access the Student Performance Prediction App.

📊 Model & Data Pipeline

🔹 Data Ingestion (Data_ingestion.py):

Reads data.csv

Splits into train (train.csv) and test (test.csv)

🔹 Data Transformation (Data_transformation.py):

Handles missing values, feature scaling, and encoding

Saves preprocessor (preprocessor.pki)

🔹 Model Training (Model_trainer.py):

Uses CatBoost for training

Saves the best model (best_model.pk)

🔹 Prediction (Predict_pipeline.py):

Loads trained model & preprocessor

Takes user input and predicts student performance

📌 Web Interface

🖥 Homepage (home.html): Input student data for prediction📊 Results Page (index.html): Displays predicted performance

🛠 Future Improvements

🚀 Add support for multiple ML models (XGBoost, Random Forest, etc.)

🚀 Improve feature engineering for better accuracy

🚀 Deploy as a Dockerized microservice

🚀 Enhance the web UI with React or Streamlit

🤝 Contributing

Contributions are welcome! To contribute:

1] Fork the repository

2] Create a new branch (feature-branch)

3] Commit your changes

4] Push and create a Pull Request

📝 License

This project is open-source under the MIT License.

🚀 Developed by Jayesh Patil📧 

Contact: 9096380075

patiljayesh6908@gmail.com

