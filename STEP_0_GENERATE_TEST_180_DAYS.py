import pandas as pd

# Define file paths
train_path = 'hull-tactical-market-prediction/train.csv'
test_path = 'hull-tactical-market-prediction/test.csv'

# Load train.csv
print("Loading train.csv...")
try:
    train_df = pd.read_csv(train_path)
    print(f"Train data loaded. Shape: {train_df.shape}")
except FileNotFoundError:
    print("Error: train.csv not found in hull-tactical-market-prediction directory.")
    exit(1)

# Ensure date_id is present
if 'date_id' not in train_df.columns:
    print("Error: date_id column not found in train.csv.")
    exit(1)

# Sort by date_id to ensure chronological order
print("Sorting train data by date_id...")
train_df = train_df.sort_values('date_id')

# Select the last 180 rows
print("Extracting last 180 days...")
test_df = train_df.tail(180).copy()
print(f"Test data shape: {test_df.shape}")

# Save to test.csv
print(f"Saving test data to {test_path}...")
test_df.to_csv(test_path, index=False)
print(f"Test data saved successfully. Shape: {test_df.shape}")
print("Test columns:", list(test_df.columns))