import os
import numpy as np
import pandas as pd
import pickle
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting the training pipeline...")

# Load dataset
try:
    df = pd.read_csv(r"notebook\data\stud.csv")  # Use raw string or os.path.join()
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

# Define features and target variable
X = df.drop(columns=["math_score"])  # Target is math_score
y = df["math_score"]

# Identify categorical and numerical features
categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
numerical_features = ["reading_score", "writing_score"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num_scaler", StandardScaler(), numerical_features),
    ("cat_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Define models
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

# Define hyperparameter grids
param_grids = {
    "RandomForest": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [10, 20, None]
    },
    "XGBoost": {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.01, 0.1, 0.2]
    },
    "LightGBM": {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.01, 0.1],
        "regressor__num_leaves": [31, 50]
    }
}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = None
best_score = float("-inf")
best_name = ""

# Start MLflow experiment
mlflow.set_experiment("Student Performance Prediction")

# Train and evaluate models
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        logging.info(f"Training {name} model...")
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("feature_selection", SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))),  # Feature Selection
            ("regressor", model)
        ])

        grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, scoring="r2", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Get best model
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        logging.info(f"Best {name} model R² Score: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Log metrics to MLflow
        mlflow.log_param("model", name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("R2_Score", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(best_estimator, name)

        if r2 > best_score:
            best_score = r2
            best_model = best_estimator
            best_name = name

mlflow.end_run()

logging.info(f"Best model: {best_name} with R² Score: {best_score:.4f}")

# Save best model
model_path = "artifacts/best_model.pkl"
os.makedirs("artifacts", exist_ok=True)
with open(model_path, "wb") as file:
    pickle.dump(best_model, file)

logging.info(f"Best model saved at {model_path}")
print(f"Training complete. Best model: {best_name} with R² Score: {best_score:.4f}")
