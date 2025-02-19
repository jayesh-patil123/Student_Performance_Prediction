import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.utils import save_object

class TrainPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'
        
    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess_data(self, df):
        try:
            X = df.drop(columns=['math_score'])  # Assuming 'math_score' is the target
            y = df['math_score']
            
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numerical_features = ["reading_score", "writing_score"]
            
            preprocessor = ColumnTransformer([
                ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('scaler', StandardScaler(), numerical_features)
            ])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_model(self, X_train, y_train):
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_artifacts(self, model, preprocessor):
        try:
            save_object(self.model_path, model)
            save_object(self.preprocessor_path, preprocessor)
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self):
        try:
            print("Loading data...")
            df = self.load_data()
            print("Preprocessing data...")
            X_train, X_test, y_train, y_test, preprocessor = self.preprocess_data(df)
            print("Training model...")
            model = self.train_model(X_train, y_train)
            print("Saving artifacts...")
            self.save_artifacts(model, preprocessor)
            print("Training pipeline completed successfully!")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_path = 'notebook\data\stud.csv'  # Update with actual dataset path
    pipeline = TrainPipeline(data_path)
    pipeline.run_pipeline()
