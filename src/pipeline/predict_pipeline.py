import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Define expected columns based on training
            expected_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                'lunch', 'test_preparation_course', 
                                'math_score', 'reading_score', 'writing_score']

            # Debugging: Print input features shape before filtering
            print("Features Shape Before Filtering:", features.shape)
            print("Feature Columns Before Filtering:", features.columns.tolist())

            # Ensure only expected columns are used
            features = features[expected_columns]

            # Debugging: Print input features shape after filtering
            print("Features Shape After Filtering:", features.shape)

            # Transform input data
            data_scaled = preprocessor.transform(features)

            # Debugging: Print shape after preprocessing
            print("Features Shape After Preprocessing:", data_scaled.shape)

            # Make predictions
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

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)