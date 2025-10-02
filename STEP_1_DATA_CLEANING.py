import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

def clean_data(train_path='hull-tactical-market-prediction/train.csv', test_path='hull-tactical-market-prediction/test.csv', impute_data=True):
    # Step 1: Load the data
    print("Loading train.csv and test.csv...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        # Optional: Explicitly remove 'is_scored' from test if present (uncomment if needed)
        # if 'is_scored' in test.columns:
        #     test = test.drop(columns=['is_scored'])
        #     print("Removed 'is_scored' from test data.")
        
        print(f"Train data loaded. Shape: {train.shape}")
        print(f"Test data loaded. Shape: {test.shape}")
        print("Train columns:", train.columns.tolist())
        print("Test columns:", test.columns.tolist())
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found in hull-tactical-market-prediction directory.")
        exit(1)

    # Step 2: Identify date column
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

    # Step 3: Drop rows with >50% null values
    null_threshold = 0.5
    print("\nDropping rows with >50% null values...")
    train_null_pct = train.isnull().sum(axis=1) / len(train.columns)
    rows_dropped_train = train[train_null_pct > null_threshold].shape[0]
    train = train[train_null_pct <= null_threshold].reset_index(drop=True)
    print(f"Train: Dropped {rows_dropped_train} rows with >50% nulls. New shape: {train.shape}")
    test_null_pct = test.isnull().sum(axis=1) / len(test.columns)
    rows_dropped_test = test[test_null_pct > null_threshold].shape[0]
    test = test[test_null_pct <= null_threshold].reset_index(drop=True)
    print(f"Test: Dropped {rows_dropped_test} rows with >50% nulls. New shape: {test.shape}")

    # Step 4: Remove first 20% of train rows
    rows_to_remove = int(0.2 * len(train))
    train = train.iloc[rows_to_remove:].reset_index(drop=True)
    print("Train shape after removing first 20%:", train.shape)

    # Step 5: Define target and feature columns
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

    # Step 6: Categorize features by type
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
        print(f"{ftype}: {len(cols)} columns")

    # Step 7: Impute missing values
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    if impute_data:
        print("\nImputing missing values...")
        impute_cols = [col for col in feature_cols]
        if feature_types['M']:
            X_train_imputed[feature_types['M']] = X_train[feature_types['M']].ffill().bfill()
            X_test_imputed[feature_types['M']] = X_test[feature_types['M']].ffill().bfill()
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
        if feature_types['I']:
            X_train_imputed[feature_types['I']] = X_train[feature_types['I']].interpolate(method='linear', limit_direction='both')
            X_test_imputed[feature_types['I']] = X_test[feature_types['I']].interpolate(method='linear', limit_direction='both')
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
        if feature_types['S']:
            X_train_imputed[feature_types['S']] = X_train[feature_types['S']].ffill().bfill()
            X_test_imputed[feature_types['S']] = X_test[feature_types['S']].ffill().bfill()
        if feature_types['MOM']:
            X_train_imputed[feature_types['MOM']] = X_train[feature_types['MOM']].ffill().bfill()
            X_test_imputed[feature_types['MOM']] = X_test[feature_types['MOM']].ffill().bfill()
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

    # Step 8: Add back auxiliary columns (exclude target for train to avoid duplication; include for test)
    test_specific_cols = ['market_forward_excess_returns']  # Explicitly exclude 'is_scored'
    aux_cols = ['forward_returns', 'risk_free_rate']
    
    # For test, add target if present
    if 'market_forward_excess_returns' in test.columns:
        aux_cols.append('market_forward_excess_returns')
    
    for col in aux_cols:
        if col in train.columns and col != target_col:  # Skip adding target to train (avoid duplication)
            X_train_imputed[col] = train[col].reindex(X_train_imputed.index)
            print(f"Added {col} back to train data")
        if col in test.columns:
            X_test_imputed[col] = test[col].reindex(X_test_imputed.index)
            print(f"Added {col} back to test data")

    # Step 9: Prepare final datasets
    train_clean = pd.concat([X_train_imputed, y_train], axis=1).dropna(subset=[target_col])
    test_clean = X_test_imputed
    
    # Safety check: Remove any duplicate columns (e.g., if target somehow duplicates)
    train_clean = train_clean.loc[:, ~train_clean.columns.duplicated()]
    test_clean = test_clean.loc[:, ~test_clean.columns.duplicated()]

    # Step 10: Save cleaned data
    train_clean.to_csv('train_cleaned.csv', index=False)
    test_clean.to_csv('test_cleaned.csv', index=False)
    print("\nTraining data saved as train_cleaned.csv")
    print("Test data saved as test_cleaned.csv")
    print("Cleaned train shape:", train_clean.shape)
    print("Cleaned test shape:", test_clean.shape)
    print("Train cleaned columns:", train_clean.columns.tolist())
    print("Test cleaned columns:", test_clean.columns.tolist())
    return train_clean, test_clean

if __name__ == "__main__":
    train_clean, test_clean = clean_data(impute_data=True)