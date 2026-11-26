from pathlib import Path
import pandas as pd

# Load raw dataset
df = pd.read_csv(Path("data/dataset.csv"))
print("Original shape:", df.shape)

# Drop saved row-index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Drop rows missing core track identifiers
required_cols = [c for c in ["track_id", "track_name", "artists"] if c in df.columns]
df = df.dropna(subset=required_cols)

# Remove duplicate tracks by track_id if present, otherwise full duplicates
if "track_id" in df.columns:
    df = df.drop_duplicates(subset=["track_id"], keep="first")
else:
    df = df.drop_duplicates()

# Save cleaned dataset separately
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "spotify_clean.csv"
df.to_csv(output_path, index=False)

print("Cleaned shape:", df.shape)
print("Saved to:", output_path)
print("\nRemaining columns:", df.columns.tolist())