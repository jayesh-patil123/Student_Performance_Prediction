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

            # Expected feature names based on the trained preprocessor
            expected_features = [
                "gender", "race_ethnicity", "parental_level_of_education", 
                "lunch", "test_preparation_course", "reading_score", "writing_score"
            ]

            # Debugging: Print input feature names
            print("Received feature columns:", list(features.columns))
            print("Expected feature columns:", expected_features)

            # Ensure only the required features are passed to the preprocessor
            features = features[expected_features]

            # Debugging: Check feature shape before transformation
            print(f"Feature shape before transformation: {features.shape}")

            # Apply transformation
            data_scaled = preprocessor.transform(features)

            # Debugging: Check transformed feature shape
            print(f"Transformed feature shape: {data_scaled.shape}")

            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    '''
    Responsible for mapping all the data points we receive in the front end to the back end.
    '''
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: int,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int) -> None:
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)

            # Debugging: Print DataFrame columns
            print("Custom DataFrame columns:", list(df.columns))

            return df

        except Exception as e:
            raise CustomException(e, sys)
