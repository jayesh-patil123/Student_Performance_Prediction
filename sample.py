import pandas as pd

files = ["stud.csv", "data.csv", "test.csv", "train.csv"]

for file in files:
    df = pd.read_csv(f"artifacts/{file}")  # Adjust the path if needed
    print(f"\n{file} columns:\n", df.columns)
