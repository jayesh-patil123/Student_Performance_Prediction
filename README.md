## End To End Machine Learning Project

## Project Structure

Student-Performance-Prediction/
│-- artifacts/               # Stores trained models, preprocessor, and other artifacts
│-- catboost_info/           # Logs and metadata from CatBoost model training
│-- data/
│   │-- data.csv             # Raw dataset
│   │-- train.csv            # Training dataset
│   │-- test.csv             # Testing dataset
│-- logs/                    # Log files for debugging and monitoring
│-- miproject.egg-info/      # Package information
│-- miruns/                  # Execution logs
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
│-- venv/                    # Virtual environment
│-- .gitignore               # Files to ignore in Git
│-- app.py                   # Main Flask application
│-- application.py           # Alternative entry point
│-- README.md                # Project documentation
│-- requirements.txt         # Python dependencies
│-- Setup.py                 # Setup script for packaging
│-- Training.log             # Logs for model training
