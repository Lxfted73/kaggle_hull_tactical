import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Optimize PyTorch for Tensor Cores (RTX 4060)
torch.set_float32_matmul_precision('high')

# Verify environment
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Sharpe ratio scoring function
def compute_sharpe_score(predictions, returns):
    allocations = np.clip(predictions, 0, 2)
    port_returns = allocations * returns
    mean_port = port_returns.mean()
    std_port = port_returns.std()
    market_std = returns.std()
    if std_port == 0:
        return 0.0
    sharpe = mean_port / std_port
    penalty = 0.5 if std_port > 1.2 * market_std else 1.0
    return sharpe * penalty

# Cumulative return
def compute_money_made(predictions, returns):
    allocations = np.clip(predictions, 0, 2)
    port_returns = allocations * returns
    return port_returns.sum()

# Load data
print("Loading train_cleaned.csv and test_cleaned.csv...")
try:
    train = pd.read_csv('train_cleaned.csv')
    test = pd.read_csv('test_cleaned.csv')
    print(f"Train data loaded. Shape: {train.shape}")
    print(f"Test data loaded. Shape: {test.shape}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"{e.filename} not found.")

# Ensure date_id is present
if 'date_id' not in train.columns or 'date_id' not in test.columns:
    raise ValueError("Both train_cleaned.csv and test_cleaned.csv must contain 'date_id' column.")

# Reindex date_id to ensure consecutive values
print("Reindexing date_id to ensure consecutive values...")
train_sorted = train.sort_values(by='date_id').reset_index(drop=True)
train_sorted['date_id'] = range(len(train_sorted))
test['original_date_id'] = test['date_id']  # Preserve original for submission
test['date_id'] = range(len(train_sorted), len(train_sorted) + len(test))
print("New train date_id range:", train_sorted['date_id'].min(), "to", train_sorted['date_id'].max())
print("New test date_id range:", test['date_id'].min(), "to", test['date_id'].max())

# Add item_id for single time series
train_sorted['item_id'] = 'SP500'
test['item_id'] = 'SP500'

# Define feature columns
target_col = 'market_forward_excess_returns'
feature_cols = [col for col in train.columns if col not in ['date_id', target_col, 'forward_returns', 'risk_free_rate', 'item_id', 'is_scored', 'original_date_id']]
print(f"Number of features: {len(feature_cols)}")

# Ensure test set has the same feature columns
missing_cols = [col for col in feature_cols if col not in test.columns]
if missing_cols:
    raise ValueError(f"Test set is missing columns: {missing_cols}")

# Prepare train and test data
X_train = train_sorted[feature_cols]
y_train = (train_sorted[target_col] > 0).astype(int)  # Binary direction
X_test = test[feature_cols]
test_ids = test['original_date_id']  # Use original date_id for submission

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

# Create PandasDataset for train and test
print("Creating PandasDataset...")
train_ds = PandasDataset.from_long_dataframe(
    dataframe=train_sorted,
    target=target_col,
    item_id="item_id",
    timestamp="date_id",
    freq="D",
    feat_dynamic_real=feature_cols
)
test_ds = PandasDataset.from_long_dataframe(
    dataframe=test,
    target=target_col,
    item_id="item_id",
    timestamp="date_id",
    freq="D",
    feat_dynamic_real=feature_cols
)

# Initialize base models
base_models = {
    'tft': TemporalFusionTransformerEstimator(
        prediction_length=1,
        context_length=30,
        hidden_dim=64,
        num_heads=4,
        freq='D',
        time_features=None,
        trainer_kwargs={'max_epochs': 10}
    ),
    'rf': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'xgb': XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[0]/class_weight_dict[1]),
    'gb': GradientBoostingClassifier(random_state=42)
}

# Initialize fold scores
fold_scores = {name: [] for name in base_models}

# Out-of-fold predictions
tscv = TimeSeriesSplit(n_splits=5)
oof_preds = np.zeros((len(X_train_scaled), len(base_models)))
print("\nGenerating out-of-fold predictions...")
for fold, (tr_idx, v_idx) in enumerate(tscv.split(X_train_scaled)):
    X_tr, y_tr = X_train_scaled.iloc[tr_idx], y_train.iloc[tr_idx]
    X_v, y_v = X_train_scaled.iloc[v_idx], y_train.iloc[v_idx]
    train_fold = train_sorted.iloc[tr_idx].reset_index(drop=True)
    train_fold['date_id'] = range(len(train_fold))  # Reindex fold to ensure consecutive date_id
    val_fold = train_sorted.iloc[v_idx].reset_index(drop=True)
    val_fold['date_id'] = range(len(val_fold))  # Reindex validation fold

    # Train non-TFT models
    for i, (name, model) in enumerate([(n, m) for n, m in base_models.items() if n != 'tft']):
        model.fit(X_tr, y_tr)
        pred_prob = model.predict_proba(X_v)[:, 1]
        oof_preds[v_idx, i] = pred_prob
        pred_class = (pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_v, pred_class)
        fold_scores[name].append(acc)

    # Train TFT
    train_ds_fold = PandasDataset.from_long_dataframe(
        dataframe=train_fold,
        target=target_col,
        item_id="item_id",
        timestamp="date_id",
        freq="D",
        feat_dynamic_real=feature_cols
    )
    tft_model = base_models['tft'].train(train_ds_fold)
    # Use val_fold for predictions to align with fold
    val_ds_fold = PandasDataset.from_long_dataframe(
        dataframe=val_fold,
        target=target_col,
        item_id="item_id",
        timestamp="date_id",
        freq="D",
        feat_dynamic_real=feature_cols
    )
    forecast_it, _ = make_evaluation_predictions(dataset=val_ds_fold, predictor=tft_model, num_samples=1)
    tft_preds = np.array([f.mean[0] for f in forecast_it])[:len(val_fold)]
    tft_preds = 1 / (1 + np.exp(-tft_preds))  # Sigmoid for binary classification
    oof_preds[v_idx, list(base_models.keys()).index('tft')] = tft_preds
    acc = accuracy_score(y_v, (tft_preds > 0.5).astype(int))
    fold_scores['tft'].append(acc)
    print(f"Fold {fold+1} completed.")

X_meta_train = pd.DataFrame(oof_preds, columns=list(base_models.keys()))

# Print model performance
print("\nModel Performance with Parameters:")
for name in fold_scores:
    avg_acc = np.mean(fold_scores[name])
    std_acc = np.std(fold_scores[name])
    print(f"{name.upper()} avg accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

# Train meta-learner
meta_model = LogisticRegression(class_weight='balanced')
meta_model.fit(X_meta_train, y_train)

# Refit models on full data
tuned_models = {
    name: model.fit(X_train_scaled, y_train) if name != 'tft' else base_models['tft'].train(train_ds)
    for name, model in base_models.items()
}

# Predict on test set
test_base_preds = np.zeros((len(X_test_scaled), len(base_models)))
for i, (name, model) in enumerate(tuned_models.items()):
    if name != 'tft':
        test_base_preds[:, i] = model.predict_proba(X_test_scaled)[:, 1]
    else:
        forecast_it, _ = make_evaluation_predictions(dataset=test_ds, predictor=model, num_samples=1)
        test_preds = np.array([f.mean[0] for f in forecast_it])[:len(X_test_scaled)]
        test_base_preds[:, i] = 1 / (1 + np.exp(-test_preds))  # Sigmoid transformation

test_base_preds = pd.DataFrame(test_base_preds, columns=list(tuned_models.keys()))
final_preds = meta_model.predict_proba(test_base_preds)[:, 1] * 2  # Scale to 0-2

# Save submission
submission = pd.DataFrame({
    'date_id': test_ids,  # Use original date_id
    'prediction': final_preds
})
submission.to_csv('submission.csv', index=False)

# Evaluation
print("\nModel Evaluation Diagnostics:")
print("Submission saved to submission.csv!")
print("Submission shape:", submission.shape)
print("Sample predictions:", final_preds[:5])
print("Prediction range:", final_preds.min(), "to", final_preds.max())
print("Ensemble std dev (diversity):", np.std(test_base_preds, axis=1).mean())
print("Final predictions std:", np.std(final_preds))

# Check target column for evaluation
if 'market_forward_excess_returns' in test.columns:
    test_returns = test['market_forward_excess_returns']
    sharpe_score = compute_sharpe_score(final_preds, test_returns)
    money_made = compute_money_made(final_preds, test_returns)
    std_port = (np.clip(final_preds, 0, 2) * test_returns).std()
    market_std = test_returns.std()
    print("\nMoney Made Criteria (Sharpe Ratio Test):")
    print(f"Sharpe Score: {sharpe_score:.4f}")
    print(f"Money Made (Cumulative Return): {money_made:.4f}")
    print(f"Volatility Penalty Applied: {'Yes (0.5)' if std_port > 1.2 * market_std else 'No (1.0)'}")

    # Directional accuracy
    pred_prob = final_preds / 2
    pred_direction = (pred_prob > 0.5).astype(int)
    actual_direction = (test['market_forward_excess_returns'] > 0).astype(int)
    correct_directions = (pred_direction == actual_direction).astype(int)
    directional_accuracy = correct_directions.mean() * 100
    print("\nDirectional Accuracy Evaluation:")
    print(f"Percentage of correct direction predictions: {directional_accuracy:.2f}%")
    print(f"Number of correct predictions: {correct_directions.sum()} out of {len(correct_directions)}")

    # Confusion matrix
    cm = confusion_matrix(actual_direction, pred_direction)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Directional)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
else:
    print("Warning: Test set does not contain 'market_forward_excess_returns'. Skipping evaluation.")