import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Verify scikit-learn version
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Flag to toggle between grid search and pre-set parameters
USE_GRID_SEARCH = False  # Keep False for quick testing

# Define pre-set parameters
best_params = {
    'gb': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50},
    'xgb': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50},
    'rf': {'max_depth': 5, 'n_estimators': 50},
    'svm': {'C': 0.1, 'gamma': 0.001}
}

# Define hyperparameter grids (for reference)
param_grids = {
    'svm': {'base_estimator__C': [0.1, 1.0, 10.0], 'base_estimator__gamma': [0.001, 0.01, 'scale']},
    'rf': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
    'xgb': {'n_estimators': [50, 100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]},
    'gb': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
}

# Custom GridSearchCV to print parameters
class VerboseGridSearchCV(GridSearchCV):
    def _fit(self, X, y, groups=None, **fit_params):
        print(f"\nTuning parameters for {self.estimator.__class__.__name__}:")
        for params in self.param_grid:
            print(f"Testing parameters: {params}")
        return super()._fit(X, y, groups, **fit_params)

# Sharpe ratio scoring function from competition (modified Sharpe with volatility penalty)
def compute_sharpe_score(predictions, returns):
    """
    Compute the competition's modified Sharpe ratio.
    - predictions: Allocations (0-2)
    - returns: market_forward_excess_returns
    Returns: Sharpe score with volatility penalty.
    """
    allocations = np.clip(predictions, 0, 2)
    port_returns = allocations * returns
    mean_port = port_returns.mean()
    std_port = port_returns.std()
    market_std = returns.std()
    
    if std_port == 0:
        return 0.0
    
    sharpe = mean_port / std_port
    # Volatility penalty: 0.5 if portfolio volatility > 1.2 * market volatility
    penalty = 0.5 if std_port > 1.2 * market_std else 1.0
    score = sharpe * penalty
    return score

# Cumulative "money made" (total return assuming $1 start)
def compute_money_made(predictions, returns):
    """
    Compute cumulative portfolio return (total "money made").
    """
    allocations = np.clip(predictions, 0, 2)
    port_returns = allocations * returns
    cumulative_return = port_returns.sum()  # Simple total return
    # For compounded: np.exp(np.cumsum(port_returns)).iloc[-1] - 1
    return cumulative_return

# Load data
print("Loading train.csv and test.csv...")
try:
    train = pd.read_csv('train_cleaned.csv')
    test = pd.read_csv('test_cleaned.csv')
    print(f"Train data loaded. Shape: {train.shape}")
    print(f"Test data loaded. Shape: {test.shape}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"{e.filename} not found in hull-tactical-market-prediction directory.")

# Ensure date_id is present
if 'date_id' not in train.columns or 'date_id' not in test.columns:
    raise ValueError("Both train.csv and test.csv must contain 'date_id' column.")

# Sort train data by date_id
print("Sorting train data by date_id...")
train_sorted = train.sort_values(by='date_id').reset_index(drop=True)

# Define feature columns
target_col = 'market_forward_excess_returns'
feature_cols = [col for col in train.columns if col not in ['date_id', target_col, 'forward_returns', 'risk_free_rate']]
print(f"Number of features: {len(feature_cols)}")

# Ensure test set has the same feature columns
missing_cols = [col for col in feature_cols if col not in test.columns]
if missing_cols:
    raise ValueError(f"Test set is missing columns: {missing_cols}")

# Prepare train and test data
X_train = train_sorted[feature_cols]
y_train = (train_sorted[target_col] > 0).astype(int)  # Binary direction: 1 if positive, 0 otherwise
X_test = test[feature_cols]
test_ids = test['date_id']

# Check class distribution
print("y_train class distribution:", pd.Series(y_train).value_counts(normalize=True))

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print("Class weights:", class_weight_dict)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# Check for NaNs
print("NaNs in X_train_scaled:", X_train_scaled.isna().sum().sum())
print("NaNs in X_test_scaled:", X_test_scaled.isna().sum().sum())
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# Initialize base models with class weights
base_models = {
    'svm': CalibratedClassifierCV(SVC(kernel='rbf', class_weight='balanced'), method='sigmoid'),
    'rf': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'xgb': XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[0]/class_weight_dict[1]),
    'gb': GradientBoostingClassifier(random_state=42)
}

# Initialize tuned models dictionary
tuned_models = {}
fold_scores = {name: [] for name in base_models}

if USE_GRID_SEARCH:
    tscv = TimeSeriesSplit(n_splits=5)
    print("\nPerforming hyperparameter tuning...")
    for name, model in base_models.items():
        print(f"\nTuning {name.upper()}...")
        grid_search = VerboseGridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='accuracy',
            cv=tscv,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train_scaled, y_train)
        tuned_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name.upper()}: {grid_search.best_params_}")
        print(f"Best accuracy for {name.upper()}: {grid_search.best_score_:.4f}")
else:
    print("\nUsing pre-set parameters:")
    for name, params in best_params.items():
        print(f"{name.upper()}: {params}")
    tuned_models = {
        'svm': CalibratedClassifierCV(SVC(kernel='rbf', class_weight='balanced', **best_params['svm']), method='sigmoid'),
        'rf': RandomForestClassifier(random_state=42, class_weight='balanced', **best_params['rf']),
        'xgb': XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[0]/class_weight_dict[1], **best_params['xgb']),
        'gb': GradientBoostingClassifier(random_state=42, **best_params['gb'])
    }

# Out-of-fold predictions for meta-learner
tscv = TimeSeriesSplit(n_splits=5)
oof_preds = np.zeros((len(X_train_scaled), len(tuned_models)))
print("\nGenerating out-of-fold predictions...")
for fold, (tr_idx, v_idx) in enumerate(tscv.split(X_train_scaled)):
    X_tr, y_tr = X_train_scaled.iloc[tr_idx], y_train.iloc[tr_idx]
    X_v, y_v = X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]
    for i, (name, model) in enumerate(tuned_models.items()):
        model.fit(X_tr, y_tr)
        pred_prob = model.predict_proba(X_v)[:, 1]
        oof_preds[v_idx, i] = pred_prob
        pred_class = (pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_v, pred_class)
        fold_scores[name].append(acc)
    print(f"Fold {fold+1} completed.")

X_meta_train = pd.DataFrame(oof_preds, columns=list(tuned_models.keys()))

# Print model performance
print("\nModel Performance with Parameters:")
for name in tuned_models:
    avg_acc = np.mean(fold_scores[name])
    std_acc = np.std(fold_scores[name])
    print(f"{name.upper()} avg accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

# Train meta-learner with class weights
meta_model = LogisticRegression(class_weight='balanced')
meta_model.fit(X_meta_train, y_train)

# Find optimal threshold using ROC curve
print("\nComputing optimal threshold...")
oof_prob = meta_model.predict_proba(X_meta_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, oof_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")

# Refit tuned models on full training data
for name, model in tuned_models.items():
    model.fit(X_train_scaled, y_train)

# Predict on test set
test_base_preds = pd.DataFrame(
    np.column_stack([model.predict_proba(X_test_scaled)[:, 1] for model in tuned_models.values()]),
    columns=list(tuned_models.keys())
)
final_preds = meta_model.predict_proba(test_base_preds)[:, 1] * 2  # Scale to 0-2

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
print("Final predictions std:", np.std(final_preds))

# Check if target column exists in test set
if 'market_forward_excess_returns' not in test.columns:
    raise ValueError("Test set does not contain 'market_forward_excess_returns' for evaluation.")

# Convert predictions to binary direction with optimal threshold (for comparison)
pred_prob = final_preds / 2
pred_direction = (pred_prob > optimal_threshold).astype(int)
actual_direction = (test['market_forward_excess_returns'] > 0).astype(int)

# Calculate directional accuracy (for comparison)
correct_directions = (pred_direction == actual_direction).astype(int)
directional_accuracy = correct_directions.mean() * 100
print("\nDirectional Accuracy Evaluation (for comparison):")
print(f"Percentage of correct direction predictions: {directional_accuracy:.2f}%")
print(f"Number of correct predictions: {correct_directions.sum()} out of {len(correct_directions)}")

# New: Compute Sharpe ratio score and money made
test_returns = test['market_forward_excess_returns']
sharpe_score = compute_sharpe_score(final_preds, test_returns)
money_made = compute_money_made(final_preds, test_returns)

print("\nMoney Made Criteria (Sharpe Ratio Test):")
print(f"Sharpe Score: {sharpe_score:.4f}")
print(f"Money Made (Cumulative Return): {money_made:.4f}")
print(f"Volatility Penalty Applied: {'Yes (0.5)' if test_returns.std() > 1.2 * test_returns.std() else 'No (1.0)'}")  # Note: penalty based on port vs market std

# Plot portfolio returns vs market returns
port_returns = np.clip(final_preds, 0, 2) * test_returns
cum_port = np.cumsum(port_returns)
cum_market = np.cumsum(test_returns)
plt.figure(figsize=(10, 5))
plt.plot(cum_port, label='Portfolio Cumulative Return')
plt.plot(cum_market, label='Market Cumulative Return')
plt.title('Cumulative Returns: Portfolio vs Market')
plt.xlabel('Day')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Confusion matrix for direction (for comparison)
cm = confusion_matrix(actual_direction, pred_direction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Directional)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()