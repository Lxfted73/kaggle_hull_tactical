import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Verify scikit-learn version
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Flag to toggle between grid search and pre-set parameters
USE_GRID_SEARCH = False  # Set to True to perform grid search, False to use pre-set parameters

# Define pre-set parameters
best_params = {
    'gb': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50},
    'xgb': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50},
    'rf': {'max_depth': 5, 'n_estimators': 50},
    'svm': {'C': 0.1, 'gamma': 0.001}
}

# Define hyperparameter grids for grid search
param_grids = {
    'svm': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': [0.001, 0.01, 0.1, 'scale']
    },
    'rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None]
    },
    'xgb': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'gb': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
}

# Custom GridSearchCV to print parameters being tuned
class VerboseGridSearchCV(GridSearchCV):
    def _fit(self, X, y, groups=None, **fit_params):
        print(f"\nTuning parameters for {self.estimator.__class__.__name__}:")
        for params in self.param_grid:
            print(f"Testing parameters: {params}")
        return super()._fit(X, y, groups, **fit_params)

# Load data
print("Loading train.csv...")
try:
    train = pd.read_csv('hull-tactical-market-prediction/train.csv')
    print(f"Train data loaded. Shape: {train.shape}")
except FileNotFoundError:
    raise FileNotFoundError("train.csv not found in hull-tactical-market-prediction directory.")

# Ensure date_id is present
if 'date_id' not in train.columns:
    raise ValueError("train.csv must contain 'date_id' column.")

# Sort by date_id and create test set (last 180 days)
print("Sorting train data by date_id...")
train_sorted = train.sort_values(by='date_id').reset_index(drop=True)
test = train_sorted.tail(180).copy()
test['is_scored'] = True  # All rows scored for public leaderboard
print(f"Test set created from last 180 days. Shape: {test.shape}")

# Define feature columns (exclude target and non-features)
target_col = 'market_forward_excess_returns'
feature_cols = [col for col in train.columns if col not in ['date_id', target_col, 'forward_returns', 'risk_free_rate']]
print(f"Number of features: {len(feature_cols)}")

# Prepare train and test data
X_train = train_sorted[feature_cols]
y_train = train_sorted[target_col]
X_test = test[feature_cols]
test_ids = test['date_id']

# Handle NaNs
print("Handling NaNs in train and test data...")
X_train = X_train.ffill().fillna(0)
X_test = X_test.ffill().fillna(0)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# Check for NaNs and data summary
print("NaNs in X_train_scaled:", X_train_scaled.isna().sum().sum())
print("NaNs in X_test_scaled:", X_test_scaled.isna().sum().sum())
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_train range:", y_train.min(), "to", y_train.max())

# Initialize base models
base_models = {
    'svm': SVR(kernel='rbf'),
    'rf': RandomForestRegressor(random_state=42),
    'xgb': XGBRegressor(random_state=42),
    'gb': GradientBoostingRegressor(random_state=42)
}

# Initialize tuned models dictionary
tuned_models = {}
fold_scores = {name: [] for name in base_models}

if USE_GRID_SEARCH:
    # Perform hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=5)
    print("\nPerforming hyperparameter tuning...")
    for name, model in base_models.items():
        print(f"\nTuning {name.upper()}...")
        grid_search = VerboseGridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train_scaled, y_train)
        tuned_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name.upper()}: {grid_search.best_params_}")
        print(f"Best RMSE for {name.upper()}: {-grid_search.best_score_:.4f}")
else:
    # Use pre-set parameters
    print("\nUsing pre-set parameters:")
    for name, params in best_params.items():
        print(f"{name.upper()}: {params}")
    tuned_models = {
        'svm': SVR(kernel='rbf', **best_params['svm']),
        'rf': RandomForestRegressor(random_state=42, **best_params['rf']),
        'xgb': XGBRegressor(random_state=42, **best_params['xgb']),
        'gb': GradientBoostingRegressor(random_state=42, **best_params['gb'])
    }

# Out-of-fold predictions for meta-learner using tuned models
tscv = TimeSeriesSplit(n_splits=5)
oof_preds = np.zeros((len(X_train_scaled), len(tuned_models)))
print("\nGenerating out-of-fold predictions...")
for fold, (tr_idx, v_idx) in enumerate(tscv.split(X_train_scaled)):
    X_tr, y_tr = X_train_scaled.iloc[tr_idx], y_train.iloc[tr_idx]
    X_v, y_v = X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]
    for i, (name, model) in enumerate(tuned_models.items()):
        model.fit(X_tr, y_tr)
        pred = model.predict(X_v)
        oof_preds[v_idx, i] = pred
        mse = mean_squared_error(y_v, pred)
        fold_scores[name].append(np.sqrt(mse))
    print(f"Fold {fold+1} completed.")

X_meta_train = pd.DataFrame(oof_preds, columns=list(tuned_models.keys()))

# Print model performance
print("\nModel Performance with Parameters:")
for name in tuned_models:
    avg_rmse = np.mean(fold_scores[name])
    std_rmse = np.std(fold_scores[name])
    print(f"{name.upper()} avg RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")

# Train meta-learner
meta_model = LinearRegression()
meta_model.fit(X_meta_train, y_train)

# Refit tuned models on full training data
for name, model in tuned_models.items():
    model.fit(X_train_scaled, y_train)

# Predict on test set
test_base_preds = pd.DataFrame(
    np.column_stack([model.predict(X_test_scaled) for model in tuned_models.values()]),
    columns=list(tuned_models.keys())
)
final_preds = meta_model.predict(test_base_preds)

# Normalize predictions
y_train_std = y_train.std()
final_preds = final_preds / np.std(final_preds) * y_train_std if np.std(final_preds) != 0 else final_preds
final_preds = np.clip(final_preds, 0, 2)

# Save submission
submission = pd.DataFrame({
    'date_id': test_ids,
    'prediction': final_preds
})
submission.to_csv('submission.csv', index=False)

# Model evaluation diagnostics
print("\nModel Evaluation Diagnostics:")
print("Submission saved to submission.csv!")
print("Submission shape:", submission.shape)
print("Sample predictions:", final_preds[:5])
print("Prediction range:", final_preds.min(), "to", final_preds.max())
print("Ensemble std dev (diversity):", np.std(test_base_preds, axis=1).mean())
print("Train target std:", y_train_std)
print("Final predictions std:", np.std(final_preds))