import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Step 1: Load and Prepare Data
data = pd.read_csv("train_cleaned.csv")  # Load preprocessed data
print("Columns in dataset:", data.columns.tolist())

# Check for date column
possible_date_cols = ['date_id', 'date', 'trading_date', 'time', 'day']
date_col = None
for col in possible_date_cols:
    if col in data.columns:
        date_col = col
        break

if date_col:
    print(f"Using {date_col} as date column (kept as numeric)")
    data = data.sort_values(date_col)  # Sort by numeric date_col
else:
    print("No date column found; assuming data is chronologically sorted")

# Identify feature columns (excluding targets, date_col, and auxiliary columns)
feature_prefixes = ['M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D']  # Match preprocessing prefixes
features = [col for col in data.columns if any(col.startswith(prefix) for prefix in feature_prefixes)]
aux_cols = ['forward_returns', 'risk_free_rate']  # Auxiliary columns from preprocessing
targets = ['market_forward_excess_returns']  # Primary target; can add 'forward_returns'

# Verify target exists
if targets[0] not in data.columns:
    raise ValueError(f"Target column '{targets[0]}' not found in dataset. Check column names.")

# Exclude auxiliary columns from features
features = [col for col in features if col not in aux_cols]

# No imputation needed (train_cleaned.csv should have no missing values after preprocessing)
print("\nMissing values in features:", data[features].isnull().sum().sum())
print("Missing values in target:", data[targets[0]].isnull().sum())

# Step 2: Pairwise Correlations (All Features)
# Pearson
pearson_corr = data[features].corr(method='pearson')
# Spearman
spearman_corr = data[features].corr(method='spearman')

# Step 3: Correlations with Target (market_forward_excess_returns)
correlations = {'Feature': [], 'Pearson': [], 'Spearman': []}
target = 'market_forward_excess_returns'

for feature in features:
    # Use lagged feature to avoid leakage
    lagged_feature = data[feature].shift(1)
    target_data = data[target]
    # Drop NaNs from lagging
    valid_data = pd.concat([lagged_feature, target_data], axis=1).dropna()
    
    # Compute correlations
    pearson_corr_val, pearson_p = pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
    spearman_corr_val, spearman_p = spearmanr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
    
    correlations['Feature'].append(feature)
    correlations['Pearson'].append(pearson_corr_val)
    correlations['Spearman'].append(spearman_corr_val)

# Convert to DataFrame
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values(by='Pearson', key=abs, ascending=False)

# Filter features with |correlation| > 0.1
selected_features = corr_df[abs(corr_df['Pearson']) > 0.1]['Feature'].tolist()
print(f"\nFeatures with |Pearson| > 0.1 with {target}:")
print(corr_df[abs(corr_df['Pearson']) > 0.1][['Feature', 'Pearson', 'Spearman']])

# Step 4: Save Plots to PDF
with PdfPages('correlation_plots.pdf') as pdf:
    # Pearson Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Heatmap (All Features)')
    pdf.savefig()  # Save to PDF
    plt.close()

    # Spearman Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(spearman_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman Correlation Heatmap (All Features)')
    pdf.savefig()  # Save to PDF
    plt.close()

    # Target Correlation Bar Plot
    plt.figure(figsize=(10, 6))
    corr_df.set_index('Feature')[['Pearson', 'Spearman']].plot(kind='bar', width=0.4)
    plt.title(f'Correlation with {target}')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0.1, color='r', linestyle='--')
    plt.axhline(y=-0.1, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()  # Save to PDF
    plt.close()

# Identify highly correlated pairs (|corr| > 0.8) for potential multicollinearity
high_corr_pairs = []
for i in range(len(pearson_corr.columns)):
    for j in range(i + 1, len(pearson_corr.columns)):
        if abs(pearson_corr.iloc[i, j]) > 0.8:
            high_corr_pairs.append((pearson_corr.columns[i], pearson_corr.columns[j], pearson_corr.iloc[i, j]))
print("Highly correlated feature pairs (|Pearson| > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

# Save correlation results
corr_df.to_csv('correlation_linear_results.csv', index=False)
print("\nCorrelation results saved to 'correlation_linear_results.csv'")
print("Plots saved to 'correlation_linear_results.pdf'")