import pickle

# Load the saved preprocessor object
preprocessor_path = 'artifacts/preprocessor.pkl'
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Print the feature names after transformation
print("Transformed feature names:", preprocessor.get_feature_names_out())
