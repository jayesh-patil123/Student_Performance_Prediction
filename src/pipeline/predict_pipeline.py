import sys
import os
import pandas as pd
import logging
from src.exception import CustomException
from src.utils import load_object

# Configure logging
logging.basicConfig(filename="logs/predict_pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class PredictPipeline:
    def __init__(self):
        """Initialize the prediction pipeline."""
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        """Loads the model and preprocessor, transforms input features, and makes predictions."""
        try:
            logging.info("Loading model and preprocessor...")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")

            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            logging.info("Transforming input data...")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions...")
            preds = model.predict(data_scaled)

            logging.info(f"Prediction completed: {preds}")
            return preds
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys)


class CustomData:
    """
    Maps user input data from the frontend to a structured DataFrame for the model.
    """
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """Converts the custom input data into a pandas DataFrame."""
        try:
            logging.info("Converting user input into DataFrame...")

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
            logging.info(f"Input DataFrame created: {df.head()}")
            return df
        except Exception as e:
            logging.error(f"Error creating DataFrame: {e}")
            raise CustomException(e, sys)
