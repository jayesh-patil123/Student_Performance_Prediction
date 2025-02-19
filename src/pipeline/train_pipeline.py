import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from src.exception import CustomException
from src.utils import save_object

class TrainPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def initiate_training(self, data: pd.DataFrame):
        try:
            # Check if the target column exists
            if "math_score" not in data.columns:
                raise ValueError("Dataset must contain 'math_score' as the target variable.")

            # Define categorical and numerical columns
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numerical_features = ["reading_score", "writing_score"]

            # Split features and target variable
            X = data.drop(columns=["math_score"])
            y = data["math_score"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Preprocessing: OneHotEncode categorical + Scale numerical
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
                ]
            )

            # Debugging: Check transformed feature shape
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            print(f"Shape after transformation (Train): {X_train_transformed.shape}")

            # Define model pipeline
            pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42))
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            error = mean_absolute_error(y_test, y_pred)
            print(f"Mean Absolute Error: {error:.2f}")

            # Ensure artifacts directory exists
            os.makedirs("artifacts", exist_ok=True)

            # Save model and preprocessor
            save_object("artifacts/model.pkl", pipeline)

            # Debugging: Check data shapes
            print("Train data shape:", X_train.shape)
            print("Test data shape:", X_test.shape)

            return error

        except Exception as e:
            raise CustomException(f"Error in training pipeline: {e}", sys)

if __name__ == "__main__":
    try:
        # Check which dataset to use
        dataset_path = "artifacts/train.csv"  # Adjust if needed

        # Load dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        data = pd.read_csv(dataset_path)
        print(f"Loaded dataset with shape: {data.shape}")

        # Initialize and train
        train_pipeline = TrainPipeline()
        error = train_pipeline.initiate_training(data)
        print(f"Training Completed with Mean Absolute Error: {error:.2f}")

    except Exception as e:
        raise CustomException(f"Error in main block: {e}", sys)
