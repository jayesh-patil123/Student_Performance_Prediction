# 🎓 Student Performance Prediction App

## 🚀 Overview
The **Student Performance Prediction App** is an end-to-end machine learning project designed to predict students' academic performance based on key factors such as study hours, attendance, previous scores, and other academic indicators. 

This project follows a structured ML pipeline, covering:
- **Data Ingestion**: Loading and preparing raw data.
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding.
- **Model Training & Evaluation**: Using **CatBoost** to achieve an **87% accuracy**.
- **Prediction & Deployment**: Serving predictions through a **Flask web interface**.

With an interactive web UI, users can input student details and receive performance predictions in real time.

---

## 📂 Project Structure
```
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
```

---

## ✨ Key Features
✅ **End-to-End ML Pipeline** – Covers data ingestion, preprocessing, model training, and deployment.  
✅ **High Accuracy (87%)** – Uses **CatBoost**, a powerful gradient boosting algorithm.  
✅ **Flask-Based Web Interface** – Enables users to input data and get predictions seamlessly.  
✅ **Logging & Exception Handling** – Ensures smooth debugging and tracking.  
✅ **Model & Preprocessor Storage** – Saves trained models for future use.  
✅ **Scalable & Extendable** – Can be improved with new features and additional ML models.  

---

## 🎯 Why This Project?
Understanding student performance trends can help educators and institutions take proactive measures to improve academic outcomes. 
This model can be used by **schools, colleges, or online learning platforms** to identify students needing intervention and provide personalized support.

---

## 🔧 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jayesh-patil123/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

### 2️⃣ Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Usage
### 1️⃣ Run the Flask App
```bash
python app.py
```
### 2️⃣ Open in Browser
Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the Student Performance Prediction App.

---

## 📊 Model & Data Pipeline
### 🔹 **Data Ingestion (Data_ingestion.py)**
- Reads `data.csv`
- Splits into train (`train.csv`) and test (`test.csv`)

### 🔹 **Data Transformation (Data_transformation.py)**
- Handles missing values, feature scaling, and encoding
- Saves preprocessor (`preprocessor.pkl`)

### 🔹 **Model Training (Model_trainer.py)**
- Uses **CatBoost** for training
- Saves the best model (`best_model.pkl`)

### 🔹 **Prediction (Predict_pipeline.py)**
- Loads trained model & preprocessor
- Takes user input and predicts student performance

---

## 📌 Web Interface
🖥 **Homepage (home.html):** Input student data for prediction  
📊 **Results Page (index.html):** Displays predicted performance  

---

## 🚀 Future Enhancements
🔹 **Add support for multiple ML models** (XGBoost, Random Forest, etc.)  
🔹 **Improve feature engineering** to enhance prediction accuracy.  
🔹 **Deploy the app as a Dockerized microservice** for scalability.  
🔹 **Enhance the web interface** using React or Streamlit for better UX.  

---

## 🤝 Contributing
Contributions are welcome! To contribute:

1️⃣ Fork the repository  
2️⃣ Create a new branch (`feature-branch`)  
3️⃣ Commit your changes  
4️⃣ Push and create a Pull Request  

---

## 🖼 Screenshots

### 📌 Homepage  
![Homepage 1 Screenshot](https://github.com/jayesh-patil123/Student_Performance_Prediction/blob/main/Screenshots/Home%20page.png)

![Homepage 2 Screenshot](https://github.com/jayesh-patil123/Student_Performance_Prediction/blob/main/Screenshots/Home%20page%20(2).png)

![Values Screenshot](https://github.com/jayesh-patil123/Student_Performance_Prediction/blob/main/Screenshots/Values.png)

![Prediction Scores Screenshot](https://github.com/jayesh-patil123/Student_Performance_Prediction/blob/main/Screenshots/score.png)

![Terminal Prediction Score Screenshot](https://github.com/jayesh-patil123/Student_Performance_Prediction/blob/main/Screenshots/terminal%20predict%20score.png)

![Terminal Training Score Screenshot](https://github.com/jayesh-patil123/Student_Performance_Prediction/blob/main/Screenshots/Training%20score.png)



## 📝 License
This project is open-source under the **MIT License**.

---

## 🚀 Developed by **Jayesh Patil** 📧
📞 **Contact:** +91 9096380075  
✉ **Email:** patiljayesh6908@gmail.com  
