import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the cleaned data
train_path = 'train_cleaned.csv'  # Path to cleaned train file
test_path = 'test_cleaned.csv'    # Path to cleaned test file (assuming it exists)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Assume data is sorted by 'date_id' for time-series handling
train = train.sort_values('date_id').reset_index(drop=True)
test = test.sort_values('date_id').reset_index(drop=True)

print("Loaded train shape:", train.shape)
print("Loaded test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())

# Step 2: Separate features and target (exclude 'date_id' and target)
target_col = 'market_forward_excess_returns'
id_col = 'date_id'

X_train = train.drop(columns=[target_col, id_col])
y_train = train[target_col]

X_test = test.drop(columns=[id_col])  # Test should have same features, no target

# Step 3: Additional Feature Engineering on Cleaned Data (brand new complex features)
# 3.1: Category-specific interactions (e.g., Volatility-adjusted Momentum if MOM exists, or proxy)
# Assuming feature types from prefixes; adapt if needed
def add_interactions(df):
    # Volatility-adjusted price (P / V)
    for i in range(1, 14):  # Assuming up to 13 for P and V
        p_col = f'P{i}' if f'P{i}' in df.columns else None
        v_col = f'V{i}' if f'V{i}' in df.columns else None
        if p_col and v_col:
            df[f'{p_col}_adj_{v_col}'] = df[p_col] / (df[v_col] + 1e-6)  # Avoid division by zero
    
    # Economic-Interest rate product (E * I for macro impact)
    for e in range(1, 21):  # E up to 20
        for i in range(1, 10):  # I up to 9
            e_col = f'E{e}' if f'E{e}' in df.columns else None
            i_col = f'I{i}' if f'I{i}' in df.columns else None
            if e_col and i_col:
                df[f'{e_col}_x_{i_col}'] = df[e_col] * df[i_col]
                break  # Limit to first few to avoid too many features
    
    # Sentiment-weighted market dynamics (S * M)
    for s in range(1, 13):  # S up to 12
        for m in range(2, 19):  # M from 2 to 18 (since M1 dropped)
            s_col = f'S{s}' if f'S{s}' in df.columns else None
            m_col = f'M{m}' if f'M{m}' in df.columns else None
            if s_col and m_col:
                df[f'{s_col}_w_{m_col}'] = df[s_col] * df[m_col]
                break
    
    return df

X_train = add_interactions(X_train)
X_test = add_interactions(X_test)  # Apply same to test

# 3.2: Exponential Moving Averages (EMA) for smoothing trends
def add_ema(df, cols, spans=[3, 5, 10]):
    for col in cols:
        for span in spans:
            df[f'{col}_ema{span}'] = df[col].ewm(span=span, adjust=False).mean()
    return df

# Apply to P*, M*, V* columns
price_cols = [col for col in X_train.columns if col.startswith('P')]
market_cols = [col for col in X_train.columns if col.startswith('M')]
vol_cols = [col for col in X_train.columns if col.startswith('V')]

X_train = add_ema(X_train, price_cols + market_cols + vol_cols)
X_test = add_ema(X_test, price_cols + market_cols + vol_cols)

# Fill any new NaNs from EMA (first few rows)
X_train = X_train.bfill()
X_test = X_test.bfill()

# 3.3: Binary feature combinations (logical OR/AND for D*)
dummy_cols = [col for col in X_train.columns if col.startswith('D')]
if dummy_cols:
    X_train['D_combined_or'] = np.any(X_train[dummy_cols] > 0.5, axis=1).astype(int)  # OR
    X_train['D_combined_and'] = np.all(X_train[dummy_cols] > 0.5, axis=1).astype(int)  # AND
    X_test['D_combined_or'] = np.any(X_test[dummy_cols] > 0.5, axis=1).astype(int)
    X_test['D_combined_and'] = np.all(X_test[dummy_cols] > 0.5, axis=1).astype(int)

print("\nFeatures after engineering:", X_train.columns.tolist())

# Step 4: Time-Series Cross-Validation Setup
tscv = TimeSeriesSplit(n_splits=5)  # 5 folds for time-series

# Step 5: Model Training with LightGBM (regression for excess returns)
params = {
    'objective': 'regression',
    'metric': 'rmse',  # Assuming RMSE as metric; change if competition uses another (e.g., MAE)
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'random_state': 42,
    'verbose': -1
}

rmse_scores = []
for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    rmse_scores.append(rmse)

print("\nCV RMSE scores:", rmse_scores)
print("Mean CV RMSE:", np.mean(rmse_scores))

# Step 6: Train on full data
full_model = lgb.LGBMRegressor(**params)
full_model.fit(X_train, y_train)

# Step 7: Predict on test
test_preds = full_model.predict(X_test)

# Step 8: Create submission file (assuming Kaggle format: date_id and predictions)
submission = pd.DataFrame({
    'date_id': test[id_col],
    'market_forward_excess_returns': test_preds
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved to 'submission.csv'")

# Step 9: Feature Importance Visualization
lgb.plot_importance(full_model, max_num_features=20, figsize=(10, 8))
plt.title('Top 20 Feature Importances')
plt.show()

# Optional: Correlation heatmap of new features (first 10 new ones)
new_cols = [col for col in X_train.columns if '_adj_' in col or '_x_' in col or '_w_' in col or '_ema' in col or 'combined' in col][:10]
if new_cols:
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_train[new_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation of New Complex Features')
    plt.show()