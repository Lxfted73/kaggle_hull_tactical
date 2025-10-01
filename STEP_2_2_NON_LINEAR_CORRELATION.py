import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_selection import mutual_info_regression
try:
    import dcor
except ImportError:
    print("Warning: dcor not installed. Falling back to Spearman correlation for distance correlation.")
    dcor = None

# Step 1: Load the cleaned data
train_path = 'train_cleaned.csv'  # Path to cleaned train file
print(f"Loading data from: {train_path}")
data = pd.read_csv(train_path)
print("Dataset loaded successfully.")
print("Columns in dataset:", data.columns.tolist())
print(f"Dataset shape: {data.shape}")

# Check for date column
possible_date_cols = ['date_id', 'date', 'trading_date', 'time', 'day']
date_col = None
for col in possible_date_cols:
    if col in data.columns:
        date_col = col
        break
if date_col:
    print(f"Using {date_col} as date column (kept as numeric)")
    print(f"Sorting data by {date_col}")
    data = data.sort_values(date_col)  # Sort by numeric date_col
else:
    print("No date column found; assuming data is chronologically sorted")

# Identify feature columns (excluding targets, date_col, and auxiliary columns)
feature_prefixes = ['M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D']
print("Feature prefixes used for filtering:", feature_prefixes)
features = [col for col in data.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]
aux_cols = ['forward_returns', 'risk_free_rate']
target = 'market_forward_excess_returns'
print(f"Target column: {target}")
print(f"Auxiliary columns: {aux_cols}")
print(f"Initial feature count: {len(features)}")

# Verify target exists
if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in dataset. Check column names.")
print(f"Target column '{target}' found in dataset.")

# Exclude auxiliary columns from features
features = [col for col in features if col not in aux_cols]
print(f"Features after excluding auxiliary columns: {features}")
print(f"Final feature count: {len(features)}")

# Check for missing values
print("\nChecking for missing values...")
print("Missing values in features:", data[features].isnull().sum().sum())
print("Missing values in target:", data[target].isnull().sum())

# Handle missing values
if data[features].isnull().sum().sum() > 0:
    print("Filling missing values in features using forward-fill and backward-fill.")
    data[features] = data[features].ffill().bfill()
if data[target].isnull().sum() > 0:
    print("Filling missing values in target using forward-fill and backward-fill.")
    data[target] = data[target].ffill().bfill()
print("Missing values after handling:", data[features].isnull().sum().sum())
print("Missing values in target after handling:", data[target].isnull().sum())

# Step 2: Compute Non-Linear Correlations
print("\nComputing non-linear correlations...")
# Initialize result dictionaries
mi_scores = {}
dcor_scores = {}
correlations = {'Feature': [], 'MI': [], 'dCor': []}

# Compute correlations with target
X = data[features]
y = data[target]
for feature in features:
    print(f"Processing feature: {feature}")
    # Use lagged feature to avoid leakage
    lagged_feature = data[feature].shift(1)
    valid_data = pd.concat([lagged_feature, y], axis=1).dropna()
    X_lagged = valid_data.iloc[:, 0]
    y_valid = valid_data.iloc[:, 1]
    print(f"Valid data points for {feature}: {len(valid_data)}")
    
    # Mutual Information
    mi_score = mutual_info_regression(X_lagged.values.reshape(-1, 1), y_valid)[0]
    mi_scores[feature] = mi_score
    correlations['Feature'].append(feature)
    correlations['MI'].append(mi_score)
    print(f"Mutual Information for {feature}: {mi_score:.4f}")
    
    # Distance Correlation (or Spearman fallback)
    if dcor:
        dcor_score = dcor.distance_correlation(X_lagged, y_valid)
        print(f"Distance Correlation for {feature}: {dcor_score:.4f}")
    else:
        dcor_score, _ = spearmanr(X_lagged, y_valid)
        print(f"Spearman Correlation (dCor fallback) for {feature}: {dcor_score:.4f}")
    dcor_scores[feature] = dcor_score
    correlations['dCor'].append(dcor_score)

# Convert to DataFrame for target correlations
print("\nCreating correlation DataFrame...")
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values(by='MI', ascending=False)
print("Top 5 features by MI:")
print(corr_df.head().to_string(index=False))

# Create pairwise correlation matrices
print("\nComputing pairwise correlation matrices...")
mi_matrix = pd.DataFrame(index=features, columns=features)
dcor_matrix = pd.DataFrame(index=features, columns=features)
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        if i <= j:
            print(f"Computing correlations between {feature1} and {feature2}")
            X1 = data[feature1].shift(1).dropna()
            X2 = data[feature2].shift(1).reindex(X1.index).dropna()
            valid_data = pd.concat([X1, X2], axis=1).dropna()
            X1_valid = valid_data.iloc[:, 0]
            X2_valid = valid_data.iloc[:, 1]
            print(f"Valid data points for pair ({feature1}, {feature2}): {len(valid_data)}")
            
            # MI
            mi_matrix.loc[feature1, feature2] = mutual_info_regression(X1_valid.values.reshape(-1, 1), X2_valid)[0]
            mi_matrix.loc[feature2, feature1] = mi_matrix.loc[feature1, feature2]
            print(f"MI between {feature1} and {feature2}: {mi_matrix.loc[feature1, feature2]:.4f}")
            
            # dCor
            if dcor:
                dcor_matrix.loc[feature1, feature2] = dcor.distance_correlation(X1_valid, X2_valid)
                print(f"dCor between {feature1} and {feature2}: {dcor_matrix.loc[feature1, feature2]:.4f}")
            else:
                dcor_matrix.loc[feature1, feature2], _ = spearmanr(X1_valid, X2_valid)
                print(f"Spearman (dCor fallback) between {feature1} and {feature2}: {dcor_matrix.loc[feature1, feature2]:.4f}")
            dcor_matrix.loc[feature2, feature1] = dcor_matrix.loc[feature1, feature2]

# Convert matrices to numeric
mi_matrix = mi_matrix.astype(float)
dcor_matrix = dcor_matrix.astype(float)
print("Pairwise correlation matrices computed.")

# Step 3: Save Plots to PDF
print("\nGenerating and saving plots to 'nonlinear_correlation_plots.pdf'...")
with PdfPages('nonlinear_correlation_plots.pdf') as pdf:
    # MI Heatmap
    print("Creating Mutual Information heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(mi_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=mi_matrix.max().max())
    plt.title('Mutual Information Heatmap (All Features)')
    pdf.savefig()
    plt.close()
    print("MI heatmap saved.")
    
    # dCor Heatmap
    print("Creating Distance Correlation heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(dcor_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Distance Correlation Heatmap (All Features)' if dcor else 'Spearman Correlation Heatmap (All Features, dCor fallback)')
    pdf.savefig()
    plt.close()
    print("dCor heatmap saved.")
    
    # Target Correlation Bar Plot
    print("Creating target correlation bar plot...")
    plt.figure(figsize=(10, 6))
    corr_df.set_index('Feature')[['MI', 'dCor']].plot(kind='bar', width=0.4)
    plt.title(f'Non-Linear Correlations with {target}')
    plt.ylabel('Score')
    plt.axhline(y=0.05, color='r', linestyle='--')  # Threshold for MI
    plt.axhline(y=0.15, color='b', linestyle='--')  # Threshold for dCor
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    print("Target correlation bar plot saved.")

# Step 4: Select Features Based on Thresholds
mi_threshold = 0.05
dcor_threshold = 0.15
print(f"\nSelecting features with MI > {mi_threshold} or dCor > {dcor_threshold}")
selected_features = corr_df[
    (corr_df['MI'] > mi_threshold) |
    (corr_df['dCor'] > dcor_threshold)
]['Feature'].tolist()
print("Selected features based on non-linear correlations:")
print(selected_features)
print(f"Number of selected features: {len(selected_features)}")

# Step 5: Save Correlation Results
print("\nSaving correlation results to 'nonlinear_correlation_results.csv'...")
corr_df.to_csv('nonlinear_correlation_results.csv', index=False)
print("Correlation results saved successfully.")
print("Plots saved to 'nonlinear_correlation_plots.pdf'")
print("Analysis complete.")