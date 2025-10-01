import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
train_path = 'hull-tactical-market-prediction/train.csv'
test_path = 'hull-tactical-market-prediction/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Check for date column
possible_date_cols = ['date_id', 'date', 'trading_date', 'time', 'day']
date_col = None
for col in possible_date_cols:
    if col in train.columns and col in test.columns:
        date_col = col
        break

if date_col:
    print(f"Using {date_col} as date column")
    train = train.sort_values(date_col)  # Sort by numeric date_col
    test = test.sort_values(date_col)
else:
    print("No date column found; assuming data is chronologically sorted")

# NEW Step: Drop rows with >50% null values (by calculation)
null_threshold = 0.5  # 50% threshold
print("\nDropping rows with >50% null values...")

# For train
train_null_pct = train.isnull().sum(axis=1) / len(train.columns)
rows_dropped_train = train[train_null_pct > null_threshold].shape[0]
train = train[train_null_pct <= null_threshold].reset_index(drop=True)
print(f"Train: Dropped {rows_dropped_train} rows with >50% nulls. New shape: {train.shape}")

# For test
test_null_pct = test.isnull().sum(axis=1) / len(test.columns)
rows_dropped_test = test[test_null_pct > null_threshold].shape[0]
test = test[test_null_pct <= null_threshold].reset_index(drop=True)
print(f"Test: Dropped {rows_dropped_test} rows with >50% nulls. New shape: {test.shape}")

# Step 2: Remove first 20% of rows (likely to have many nulls) - Now applied after null-based dropping
rows_to_remove = int(0.2 * len(train))
train = train.iloc[rows_to_remove:].reset_index(drop=True)

print("Train shape after removing first 20%:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())
print("\nTest columns:", test.columns.tolist())
print("\nTarget column:", 'market_forward_excess_returns')

# Step 3: Drop columns with >50% missing values
columns_to_drop = ['E7', 'M1', 'M13', 'M14', 'S3', 'V10']  # As identified
print(f"\nDropping {len(columns_to_drop)} columns with >50% missing values:", columns_to_drop)

train = train.drop(columns=columns_to_drop, errors='ignore')
test = test.drop(columns=columns_to_drop, errors='ignore')

# Separate features and target
target_col = 'market_forward_excess_returns'
y_train = train[target_col]

# Exclude non-feature columns from X_train, but keep date_col
exclude_cols = [target_col, 'forward_returns', 'risk_free_rate'] + ([date_col] if date_col else [])
feature_cols = [col for col in train.columns if col not in exclude_cols]
X_train = train[feature_cols + ([date_col] if date_col else [])]  # Include date_col
# Align test features (only include columns present in test)
common_cols = [col for col in feature_cols if col in test.columns]
X_test = test[common_cols + ([date_col] if date_col else [])]  # Include date_col

print("\nCommon feature columns (excluding date):", common_cols)

# Step 4: Categorize features by type (excluding date_col)
feature_types = {
    'M': [col for col in common_cols if col.startswith('M')],  # Market Dynamics/Technical
    'E': [col for col in common_cols if col.startswith('E')],  # Macro Economic
    'I': [col for col in common_cols if col.startswith('I')],  # Interest Rate
    'P': [col for col in common_cols if col.startswith('P')],  # Price/Valuation
    'V': [col for col in common_cols if col.startswith('V')],  # Volatility
    'S': [col for col in common_cols if col.startswith('S')],  # Sentiment
    'MOM': [col for col in common_cols if col.startswith('MOM')],  # Momentum
    'D': [col for col in common_cols if col.startswith('D')]   # Dummy/Binary
}

print("\nFeature types identified:")
for ftype, cols in feature_types.items():
    print(f"{ftype}: {len(cols)} columns", cols)

# Step 5: Handle Missing Values by Feature Type
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

# Only impute feature columns (exclude date_col)
impute_cols = [col for col in X_train.columns if col != date_col]

# M* (Market Dynamics/Technical): Forward fill
if feature_types['M']:
    X_train_imputed[feature_types['M']] = X_train[feature_types['M']].ffill().bfill()
    X_test_imputed[feature_types['M']] = X_test[feature_types['M']].ffill().bfill()

# E* (Macro Economic): Median imputation
if feature_types['E']:
    median_imputer = SimpleImputer(strategy='median')
    X_train_imputed[feature_types['E']] = pd.DataFrame(
        median_imputer.fit_transform(X_train[feature_types['E']]),
        columns=feature_types['E'], index=X_train.index
    )
    X_test_imputed[feature_types['E']] = pd.DataFrame(
        median_imputer.transform(X_test[feature_types['E']]),
        columns=feature_types['E'], index=X_test.index
    )

# I* (Interest Rate): Linear interpolation
if feature_types['I']:
    X_train_imputed[feature_types['I']] = X_train[feature_types['I']].interpolate(method='linear', limit_direction='both')
    X_test_imputed[feature_types['I']] = X_test[feature_types['I']].interpolate(method='linear', limit_direction='both')

# P* (Price/Valuation): KNN imputation
if feature_types['P']:
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_imputed[feature_types['P']] = pd.DataFrame(
        knn_imputer.fit_transform(X_train[feature_types['P']]),
        columns=feature_types['P'], index=X_train.index
    )
    X_test_imputed[feature_types['P']] = pd.DataFrame(
        knn_imputer.transform(X_test[feature_types['P']]),
        columns=feature_types['P'], index=X_test.index
    )

# V* (Volatility): Mean imputation
if feature_types['V']:
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_imputed[feature_types['V']] = pd.DataFrame(
        mean_imputer.fit_transform(X_train[feature_types['V']]),
        columns=feature_types['V'], index=X_train.index
    )
    X_test_imputed[feature_types['V']] = pd.DataFrame(
        mean_imputer.transform(X_test[feature_types['V']]),
        columns=feature_types['V'], index=X_test.index
    )

# S* (Sentiment): Forward fill
if feature_types['S']:
    X_train_imputed[feature_types['S']] = X_train[feature_types['S']].ffill().bfill()
    X_test_imputed[feature_types['S']] = X_test[feature_types['S']].ffill().bfill()

# MOM* (Momentum): Forward fill
if feature_types['MOM']:
    X_train_imputed[feature_types['MOM']] = X_train[feature_types['MOM']].ffill().bfill()
    X_test_imputed[feature_types['MOM']] = X_test[feature_types['MOM']].ffill().bfill()

# D* (Dummy/Binary): Mode imputation
if feature_types['D']:
    mode_imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed[feature_types['D']] = pd.DataFrame(
        mode_imputer.fit_transform(X_train[feature_types['D']]),
        columns=feature_types['D'], index=X_train.index
    )
    X_test_imputed[feature_types['D']] = pd.DataFrame(
        mode_imputer.transform(X_test[feature_types['D']]),
        columns=feature_types['D'], index=X_test.index
    )

# Add back excluded columns ('forward_returns', 'risk_free_rate') to X_train_imputed for completeness
excluded_aux_cols = ['forward_returns', 'risk_free_rate']
for col in excluded_aux_cols:
    if col in train.columns:
        X_train_imputed[col] = train[col].reindex(X_train_imputed.index)
        print(f"Added {col} back to train data (shape after reindex: {X_train_imputed[col].shape})")

# Keep date_col in imputed data (no imputation needed)
if date_col:
    X_train_imputed[date_col] = train[date_col].reindex(X_train_imputed.index)
    X_test_imputed[date_col] = test[date_col].reindex(X_test_imputed.index)

# Handle remaining NaNs in target (drop rows)
train_clean = pd.concat([X_train_imputed, y_train], axis=1).dropna()
X_train_clean = train_clean.drop(columns=[target_col])
y_train_clean = train_clean[target_col]

# Step 6: Initial Exploration
print("\nMissing values in train features after imputation:", X_train_imputed[impute_cols].isnull().sum().sum())
print("\nMissing values in target:", y_train_clean.isnull().sum())
print("\nDuplicate rows in train:", train_clean.duplicated().sum())

# Visualize target distribution
plt.figure(figsize=(8, 4))
sns.histplot(y_train_clean, kde=True)
plt.title('Distribution of Market Forward Excess Returns')
plt.show()

# Correlation heatmap for first 10 feature columns (excluding date_col and aux cols)
corr_cols = [col for col in X_train_clean.columns if col not in ([date_col] if date_col else []) and col not in excluded_aux_cols][:10]
plt.figure(figsize=(10, 8))
sns.heatmap(X_train_clean[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (First 10 Features)')
plt.show()

# Step 7: Detect and Handle Outliers
# def remove_outliers_iqr(df, col):
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     return df[(df[col] >= lower) & (df[col] <= upper)]

# # Apply to target
# train_no_outliers = remove_outliers_iqr(train_clean, target_col)
# X_train_no_out = train_no_outliers.drop(columns=[target_col])
# y_train_no_out = train_no_outliers[target_col]

# # Clip outliers in features (exclude date_col and aux_cols)
# clip_cols = [c for c in X_train_clean.columns if c != date_col and c not in excluded_aux_cols]
# for col in clip_cols:
#     Q1 = X_train_clean[col].quantile(0.25)
#     Q3 = X_train_clean[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     X_train_clean[col] = np.clip(X_train_clean[col], lower, upper)
#     if col in X_test_imputed.columns:
#         X_test_imputed[col] = np.clip(X_test_imputed[col], lower, upper)

# # Step 8: Feature Scaling (exclude date_col and aux_cols)
# scaler = StandardScaler()
# scale_cols = [col for col in X_train_clean.columns if col != date_col and col not in excluded_aux_cols]
# X_train_scaled_temp = scaler.fit_transform(X_train_clean[scale_cols])
# X_test_scaled_temp = scaler.transform(X_test_imputed[scale_cols])

# # Create scaled DataFrames
# X_train_scaled = pd.DataFrame(X_train_scaled_temp, columns=scale_cols, index=X_train_clean.index)
# X_test_scaled = pd.DataFrame(X_test_scaled_temp, columns=scale_cols, index=X_test_imputed.index)

# # Reattach date_col and aux_cols to scaled data (aux_cols only for train)
# if date_col:
#     X_train_scaled[date_col] = X_train_clean[date_col]
#     X_test_scaled[date_col] = X_test_imputed[date_col]

# for col in excluded_aux_cols:
#     if col in X_train_clean.columns:
#         X_train_scaled[col] = X_train_clean[col]

# Step 9: Save cleaned data
train_clean.to_csv('train_cleaned.csv', index=False)
X_test_imputed.to_csv('test_cleaned.csv', index=False)

print("\nCleaning complete! Cleaned train shape:", X_train_clean.shape)
# print("Outlier-removed train shape:", X_train_no