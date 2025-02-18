import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Expected feature names based on training
            expected_features = [
                "gender", "race_ethnicity", "parental_level_of_education", 
                "lunch", "test_preparation_course", "reading_score", "writing_score"
            ]

            # Debugging: Print received feature names
            print("Received feature columns:", list(features.columns))

            # Drop any extra columns that are not in expected_features
            features = features[expected_features]  

            print("Received feature columns:", list(features.columns))
            print(f"Feature shape before transformation: {features.shape}")

            # Debugging: Check feature shape before transformation
            print(f"Feature shape before transformation: {features.shape}")

            # Apply transformation
            data_scaled = preprocessor.transform(features)

            # Debugging: Check transformed feature shape
            print(f"Transformed feature shape: {data_scaled.shape}")
            print(f"Expected transformed feature shape: {preprocessor.transform(features[:1]).shape}")

            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)
