import os
import sys
import json
import pickle
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
from src.utils import save_object

# Configure logging
logging.basicConfig(filename="logs/train_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class TrainPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.artifact_dir = "artifacts"
        os.makedirs(self.artifact_dir, exist_ok=True)

    def load_data(self):
        """Loads dataset from CSV file and checks for missing values."""
        try:
            logging.info("Loading dataset...")
            df = pd.read_csv(self.data_path)

            if df.isnull().sum().any():
                raise ValueError("Dataset contains missing values. Please handle them before training.")

            logging.info(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        """Preprocesses data including scaling, encoding, and feature selection."""
        try:
            logging.info("Starting data preprocessing...")

            # Splitting features and target
            X = df.drop(columns=['math_score'])
            y = df['math_score']

            # Defining categorical & numerical features
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numerical_features = ["reading_score", "writing_score"]

            # Preprocessing pipeline
            preprocessor = ColumnTransformer([
                ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('scaler', StandardScaler(), numerical_features)
            ])

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply transformations
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Feature selection
            selector = SelectKBest(score_func=f_regression, k="all")  # Change 'all' to a specific number if needed
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)

            logging.info("Data preprocessing completed successfully.")
            return X_train_selected, X_test_selected, y_train, y_test, preprocessor
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train):
        """Trains a RandomForest model with hyperparameter tuning."""
        try:
            logging.info("Starting model training with hyperparameter tuning...")

            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            logging.info(f"Best Model Parameters: {grid_search.best_params_}")

            return best_model
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluates the model and logs performance metrics."""
        try:
            logging.info("Evaluating model performance...")
            y_pred = model.predict(X_test)

            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE (fixed)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model Evaluation -> R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            return r2, mae, rmse
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise CustomException(e, sys)

    def save_evaluation_report(self, report):
        """Saves model evaluation metrics to a JSON file."""
        try:
            report_path = os.path.join(self.artifact_dir, "model_evaluation.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            logging.info(f"Model evaluation report saved at: {report_path}")
        except Exception as e:
            logging.error(f"Error saving evaluation report: {e}")
            raise CustomException(e, sys)

    def save_artifacts(self, model, preprocessor):
        """Saves trained model and preprocessor."""
        try:
            model_path = os.path.join(self.artifact_dir, "model.pkl")
            preprocessor_path = os.path.join(self.artifact_dir, "preprocessor.pkl")

            save_object(model_path, model)
            save_object(preprocessor_path, preprocessor)

            logging.info(f"Artifacts saved successfully: {model_path}, {preprocessor_path}")
        except Exception as e:
            logging.error(f"Error saving artifacts: {e}")
            raise CustomException(e, sys)

    def run_pipeline(self):
        """Executes the complete ML training pipeline."""
        try:
            logging.info("Starting Training Pipeline...")

            df = self.load_data()
            X_train, X_test, y_train, y_test, preprocessor = self.preprocess_data(df)
            model = self.train_model(X_train, y_train)

            # Evaluate the model
            r2, mae, rmse = self.evaluate_model(model, X_test, y_test)

            # Save evaluation results
            report = {"R2 Score": r2, "MAE": mae, "RMSE": rmse}
            self.save_evaluation_report(report)

            # Save model and preprocessor
            self.save_artifacts(model, preprocessor)

            logging.info("Training pipeline completed successfully!")
        except Exception as e:
            logging.critical(f"Pipeline failed: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_path = os.path.join("notebook", "data", "stud.csv")  # Fixed path issue
    pipeline = TrainPipeline(data_path)
    pipeline.run_pipeline()
