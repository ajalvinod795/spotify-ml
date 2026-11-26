from pathlib import Path
import pandas as pd

csv_path = Path("data/dataset.csv")
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nColumn names and types:")
print(df.dtypes)

print("\nUnique values per column:")
print(df.nunique().sort_values())