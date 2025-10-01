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

# Step 2: Remove first 20% of rows (likely to have many nulls)
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
# Exclude non-feature columns from X_train
feature_cols = [col for col in train.columns if col not in [target_col, 'forward_returns', 'risk_free_rate', 'date_id']]
X_train = train[feature_cols]

# Align test features (only include columns present in test)
common_cols = [col for col in feature_cols if col in test.columns]
X_test = test[common_cols]

print("\nCommon feature columns:", common_cols)

# Step 4: Categorize features by type
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

# Handle remaining NaNs in target (drop rows)
train_clean = pd.concat([X_train_imputed, y_train], axis=1).dropna()
X_train_clean = train_clean.drop(columns=[target_col])
y_train_clean = train_clean[target_col]

# Step 6: Initial Exploration
print("\nMissing values in train features after imputation:", X_train_imputed.isnull().sum().sum())
print("\nMissing values in target:", y_train_clean.isnull().sum())
print("\nDuplicate rows in train:", train_clean.duplicated().sum())

# Visualize target distribution
plt.figure(figsize=(8, 4))
sns.histplot(y_train_clean, kde=True)
plt.title('Distribution of Market Forward Excess Returns')
plt.show()

# Correlation heatmap for first 10 features
plt.figure(figsize=(10, 8))
sns.heatmap(X_train_clean.iloc[:, :10].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (First 10 Features)')
plt.show()

# Step 7: Detect and Handle Outliers
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

# Apply to target
train_no_outliers = remove_outliers_iqr(train_clean, target_col)
X_train_no_out = train_no_outliers.drop(columns=[target_col])
y_train_no_out = train_no_outliers[target_col]

# Clip outliers in features
for col in X_train_clean.columns:
    Q1 = X_train_clean[col].quantile(0.25)
    Q3 = X_train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X_train_clean[col] = np.clip(X_train_clean[col], lower, upper)
    X_test_imputed[col] = np.clip(X_test_imputed[col], lower, upper)

# Step 8: Feature Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_clean), columns=X_train_clean.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

# Step 9: Save cleaned data
train_clean.to_csv('train_cleaned.csv', index=False)
X_test_imputed.to_csv('test_cleaned.csv', index=False)

print("\nCleaning complete! Cleaned train shape:", X_train_clean.shape)
print("Outlier-removed train shape:", X_train_no_out.shape)