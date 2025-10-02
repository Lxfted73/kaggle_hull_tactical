import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(train_path='hull-tactical-market-prediction/train.csv', test_path='hull-tactical-market-prediction/test.csv', impute_data=True):
    # Step 1: Load the data
    print("Loading train.csv and test.csv...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print(f"Train data loaded. Shape: {train.shape}")
        print(f"Test data loaded. Shape: {test.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found in hull-tactical-market-prediction directory.")
        exit(1)

    # Check for date column
    possible_date_cols = ['date_id', 'date', 'trading_date', 'time', 'day']
    date_col = None
    for col in possible_date_cols:
        if col in train.columns and col in test.columns:
            date_col = col
            break

    if date_col:
        print(f"Using {date_col} as date column")
        train = train.sort_values(date_col)
        test = test.sort_values(date_col)
    else:
        print("No date column found; assuming data is chronologically sorted")

    # Step 2: Drop rows with >50% null values
    null_threshold = 0.5  # 50% threshold
    print("\nDropping rows with >50% null values...")

    train_null_pct = train.isnull().sum(axis=1) / len(train.columns)
    rows_dropped_train = train[train_null_pct > null_threshold].shape[0]
    train = train[train_null_pct <= null_threshold].reset_index(drop=True)
    print(f"Train: Dropped {rows_dropped_train} rows with >50% nulls. New shape: {train.shape}")

    test_null_pct = test.isnull().sum(axis=1) / len(test.columns)
    rows_dropped_test = test[test_null_pct > null_threshold].shape[0]
    test = test[test_null_pct <= null_threshold].reset_index(drop=True)
    print(f"Test: Dropped {rows_dropped_test} rows with >50% nulls. New shape: {test.shape}")

    # Step 3: Remove first 20% of rows from train (likely to have many nulls)
    rows_to_remove = int(0.2 * len(train))
    train = train.iloc[rows_to_remove:].reset_index(drop=True)
    print("Train shape after removing first 20%:", train.shape)
    print("\nTrain columns:", train.columns.tolist())
    print("\nTarget column:", 'market_forward_excess_returns')

    # Step 4: Retain all columns (no dropping of columns)
    print("\nKeeping all columns from original train.csv and test.csv...")

    # Separate features and target for train
    target_col = 'market_forward_excess_returns'
    if target_col not in train.columns:
        print(f"Error: Target column '{target_col}' not found in train data.")
        exit(1)
    y_train = train[target_col]
    exclude_cols = [target_col, date_col] if date_col else [target_col]
    feature_cols = [col for col in train.columns if col not in exclude_cols]
    
    # Ensure test set has the same feature columns
    missing_cols = [col for col in feature_cols if col not in test.columns]
    if missing_cols:
        print(f"Warning: Test set is missing columns: {missing_cols}. Filling with NaNs.")
        for col in missing_cols:
            test[col] = np.nan

    X_train = train[feature_cols + ([date_col] if date_col else [])]
    X_test = test[feature_cols + ([date_col] if date_col else [])]

    # Step 5: Categorize features by type (excluding date_col)
    feature_types = {
        'M': [col for col in feature_cols if col.startswith('M')],
        'E': [col for col in feature_cols if col.startswith('E')],
        'I': [col for col in feature_cols if col.startswith('I')],
        'P': [col for col in feature_cols if col.startswith('P')],
        'V': [col for col in feature_cols if col.startswith('V')],
        'S': [col for col in feature_cols if col.startswith('S')],
        'MOM': [col for col in feature_cols if col.startswith('MOM')],
        'D': [col for col in feature_cols if col.startswith('D')]
    }

    print("\nFeature types identified:")
    for ftype, cols in feature_types.items():
        print(f"{ftype}: {len(cols)} columns", cols)

    # Step 6: Handle Missing Values by Feature Type (if impute_data is True)
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    if impute_data:
        print("\nImputing missing values...")
        # Only impute feature columns (exclude date_col)
        impute_cols = [col for col in feature_cols]

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
    else:
        print("\nSkipping imputation, keeping null values...")
        impute_cols = [col for col in X_train.columns if col != date_col]

    # Add back auxiliary columns to X_train_imputed and X_test_imputed
    excluded_aux_cols = ['forward_returns', 'risk_free_rate']
    for col in excluded_aux_cols:
        if col in train.columns:
            X_train_imputed[col] = train[col].reindex(X_train_imputed.index)
            print(f"Added {col} back to train data (shape after reindex: {X_train_imputed[col].shape})")
        if col in test.columns:
            X_test_imputed[col] = test[col].reindex(X_test_imputed.index)
            print(f"Added {col} back to test data (shape after reindex: {X_test_imputed[col].shape})")

    # Keep date_col in imputed data (no imputation needed)
    if date_col:
        X_train_imputed[date_col] = train[date_col].reindex(X_train_imputed.index)
        X_test_imputed[date_col] = test[date_col].reindex(X_test_imputed.index)

    # Handle remaining NaNs in train target (drop rows)
    train_clean = pd.concat([X_train_imputed, y_train], axis=1).dropna(subset=[target_col])
    X_train_clean = train_clean.drop(columns=[target_col])
    y_train_clean = train_clean[target_col]

    # For test data, keep all rows (no target to drop on)
    test_clean = X_test_imputed

    # Step 7: Initial Exploration
    try:
        print("\nMissing values in train features after processing:", X_train_imputed[impute_cols].isnull().sum().sum())
        print("\nMissing values in test features after processing:", X_test_imputed[impute_cols].isnull().sum().sum())
        print("\nMissing values in train target:", y_train_clean.isnull().sum())
        print("\nDuplicate rows in train:", train_clean.duplicated().sum())
        print("\nDuplicate rows in test:", test_clean.duplicated().sum())
    except Exception as e:
        print(f"Error during exploration: {e}")
        print("Skipping detailed exploration due to null values or column issues.")

    # Visualize target distribution (train only)
    try:
        plt.figure(figsize=(8, 4))
        sns.histplot(y_train_clean, kde=True)
        plt.title('Distribution of Market Forward Excess Returns')
        plt.show()
    except Exception as e:
        print(f"Error plotting target distribution: {e}")

    # Correlation heatmap for first 10 feature columns (excluding date_col and aux cols)
    try:
        corr_cols = [col for col in X_train_clean.columns if col not in ([date_col] if date_col else []) and col not in excluded_aux_cols][:10]
        if len(corr_cols) > 1:  # Ensure there are enough columns for correlation
            plt.figure(figsize=(10, 8))
            sns.heatmap(X_train_clean[corr_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap (First 10 Features)')
            plt.show()
        else:
            print("Not enough valid columns for correlation heatmap.")
    except Exception as e:
        print(f"Error plotting correlation heatmap: {e}")

    # Step 8: Save cleaned data
    train_clean.to_csv('train_cleaned.csv', index=False)
    test_clean.to_csv('test_cleaned.csv', index=False)
    print("\nTraining data saved as train_cleaned.csv")
    print("Test data saved as test_cleaned.csv")

    print("\nCleaning complete! Cleaned train shape:", X_train_clean.shape)
    print("Cleaned test shape:", test_clean.shape)
    return train_clean, X_train_clean, y_train_clean, test_clean

if __name__ == "__main__":
    # Example usage: Set impute_data to True or False
    train_clean, X_train_clean, y_train_clean, test_clean = clean_data(impute_data=True)