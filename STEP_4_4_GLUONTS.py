'''
This script implements a time series forecasting pipeline using the GluonTS library and a Temporal Fusion Transformer (TFT)
model to predict market forward excess returns for the S&P 500. It begins by loading pre-cleaned training and testing
datasets from CSV files, reindexes the 'date_id' column to a synthetic daily datetime format to ensure uniform temporal
spacing suitable for time series modeling, and adds a dummy 'item_id' to treat the data as a single multivariate time series. 
Feature columns are dynamically identified, excluding targets and metadata, and PandasDatasets are created for both train and 
test sets with dynamic real-valued features. The TFT estimator is configured with a prediction horizon matching the test set 
length (180 days), a short context window, and other hyperparameters for efficient training on daily frequency without 
relying on built-in time features due to the synthetic dates. The model is trained on the training dataset, and
predictions are generated on the test set using multiple samples for probabilistic forecasting, followed by transforming the 
raw outputs via sigmoid scaling to fit the required [0, 2] allocation range. A submission CSV is 
generated with original date_ids and predictions. If ground truth labels are available in the test set, 
the script evaluates performance using GluonTS Evaluator for standard metrics plus custom metrics like a penalized Sharpe 
ratio (accounting for volatility relative to market std), cumulative portfolio returns, directional accuracy for binary 
return signs, and visualizes a confusion matrix for directional predictions. Environment checks for PyTorch and CUDA ensure 
GPU acceleration, and various print statements provide diagnostics for debugging and verification.
'''

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator  # Added for tutorial alignment
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

# Reindex date_id to a DateTime index
print("Reindexing date_id to a DateTime index...")
train_sorted = train.sort_values(by='date_id').reset_index(drop=True)
test['original_date_id'] = test['date_id']  # Preserve original for submission
start_date = pd.to_datetime('1970-01-01')
train_sorted['date_id'] = start_date + pd.to_timedelta(train_sorted.index, unit='D')
test['date_id'] = start_date + pd.to_timedelta(range(len(train_sorted), len(train_sorted) + len(test)), unit='D')
print("New train date_id range:", train_sorted['date_id'].min(), "to", train_sorted['date_id'].max())
print("New test date_id range:", test['date_id'].min(), "to", test['date_id'].max())

# Verify uniform spacing
print("Train date_id spacing:", train_sorted['date_id'].diff().dropna().dt.days.unique())
print("Test date_id spacing:", test['date_id'].diff().dropna().dt.days.unique())

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


# Create PandasDataset
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
    target=target_col if 'market_forward_excess_returns' in test.columns else None,  # Handle if no target in test
    item_id="item_id",
    timestamp="date_id",
    freq="D",
    feat_dynamic_real=feature_cols
)

# Configure TFT Estimator with PyTorch Lightning kwargs
tft_estimator = TemporalFusionTransformerEstimator(
    prediction_length=180,  # Matches test length
    context_length=30,  # Short window
    hidden_dim=64,
    num_heads=4,
    freq='D',
    time_features=None,  # No built-in due to synthetic dates
    trainer_kwargs={  # Dict passed to Lightning Trainer
        "max_epochs": 1,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1 if torch.cuda.is_available() else None,
        "enable_progress_bar": True,  # For logging
        "logger": True,  # Optional: For monitoring
    }
)

# Train TFT
print("Training TFT model...")
tft_predictor = tft_estimator.train(train_ds)  # Returns Predictor, as in tutorial

# Predict on test set (tutorial-style: probabilistic with samples)
print("Generating test predictions...")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=tft_predictor,
    num_samples=100  # For smooth uncertainty; can reduce to 10 for speed
)
forecasts = list(forecast_it)  # List of 1 forecast (single series)
tss = list(ts_it)  # List of 1 time series (GT)

# Extract full prediction horizon (len=180) from the single forecast
# Use .median (per warning; .mean may not be stored in TFT)
test_preds = forecasts[0].median  # 1D array: shape (prediction_length,)
test_preds = 1 / (1 + np.exp(-test_preds)) * 2  # Sigmoid scale to [0, 2]

# Verify prediction length
print("Length of test_preds:", len(test_preds))
if len(test_preds) != len(test):
    raise ValueError(f"Prediction length {len(test_preds)} does not match test set length {len(test)}")

# Optional: Plot forecast fan (tutorial-inspired visualization)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
tss[0][- (len(test_preds) + 30) :].plot(label="Historical + GT", color="black", linewidth=2)  # Last context + test
forecasts[0].plot(color="blue")  # Fan with 50%/90% quantiles
ax.set_title("TFT Forecast: S&P 500 Excess Returns (with Uncertainty)")
ax.set_ylabel("Excess Returns")
ax.legend()
plt.tight_layout()
plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')  # Save as PNG for easy sharing
plt.show()

# Save submission
submission = pd.DataFrame({
    'date_id': test['original_date_id'],
    'prediction': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Saved: submission.csv")

# Evaluation (as before, but with full preds)
print("\nModel Evaluation Diagnostics:")
print("Submission saved to submission.csv!")
print("Submission shape:", submission.shape)
print("Sample predictions:", test_preds[:5])
print("Prediction range:", test_preds.min(), "to", test_preds.max())
print("Predictions std:", np.std(test_preds))

# Save predictions diagnostics to CSV
diagnostics_data = {
    'Metric': ['Submission Shape Rows', 'Submission Shape Cols', 'Sample Pred 1', 'Sample Pred 2', 'Sample Pred 3', 
               'Sample Pred 4', 'Sample Pred 5', 'Min Prediction', 'Max Prediction', 'Std Predictions'],
    'Value': [submission.shape[0], submission.shape[1], *test_preds[:5], test_preds.min(), test_preds.max(), np.std(test_preds)]
}
diagnostics_df = pd.DataFrame(diagnostics_data)
diagnostics_df.to_csv('predictions_diagnostics.csv', index=False)
print("Saved: predictions_diagnostics.csv")

if 'market_forward_excess_returns' in test.columns:
    # GluonTS Evaluator (tutorial-style; uses full horizon for metrics)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print("\nGluonTS Aggregate Metrics:")
    metrics_df = pd.DataFrame.from_dict(agg_metrics, orient='index', columns=['Value'])
    print(metrics_df.round(4).to_string())

    # Save GluonTS metrics to CSVs
    metrics_df.to_csv('gluonts_aggregate_metrics.csv')
    print("Saved: gluonts_aggregate_metrics.csv")
    item_metrics.to_csv('gluonts_item_metrics.csv')
    print("Saved: gluonts_item_metrics.csv")

    # Custom metrics (using point predictions from median)
    test_returns = test['market_forward_excess_returns'].values  # Align shapes
    sharpe_score = compute_sharpe_score(test_preds, test_returns)
    money_made = compute_money_made(test_preds, test_returns)
    std_port = (np.clip(test_preds, 0, 2) * test_returns).std()
    market_std = test_returns.std()
    print("\nMoney Made Criteria (Sharpe Ratio Test):")
    print(f"Sharpe Score: {sharpe_score:.4f}")
    print(f"Money Made (Cumulative Return): {money_made:.4f}")
    print(f"Volatility Penalty Applied: {'Yes (0.5)' if std_port > 1.2 * market_std else 'No (1.0)'}")

    # Directional accuracy
    pred_prob = test_preds / 2
    pred_direction = (pred_prob > 0.5).astype(int)
    actual_direction = (test_returns > 0).astype(int)
    directional_accuracy = (pred_direction == actual_direction).mean() * 100
    num_correct = (pred_direction == actual_direction).sum()
    total_samples = len(pred_direction)
    print("\nDirectional Accuracy Evaluation:")
    print(f"Percentage of correct direction predictions: {directional_accuracy:.2f}%")
    print(f"Number of correct predictions: {num_correct} out of {total_samples}")

    # Save custom metrics to CSV
    custom_metrics_data = {
        'Metric': ['Sharpe Score', 'Money Made (Cumulative Return)', 'Volatility Penalty Applied', 
                   'Directional Accuracy (%)', 'Number of Correct Predictions', 'Total Samples'],
        'Value': [sharpe_score, money_made, 'Yes (0.5)' if std_port > 1.2 * market_std else 'No (1.0)', 
                  directional_accuracy, num_correct, total_samples]
    }
    custom_metrics_df = pd.DataFrame(custom_metrics_data)
    custom_metrics_df.to_csv('custom_metrics.csv', index=False)
    print("Saved: custom_metrics.csv")

    # Confusion matrix
    cm = confusion_matrix(actual_direction, pred_direction)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Directional)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # Save as PNG
    plt.show()

    # Save confusion matrix as CSV (rows: Actual, cols: Predicted)
    cm_df = pd.DataFrame(cm, 
                         index=['Actual Negative', 'Actual Positive'], 
                         columns=['Predicted Negative', 'Predicted Positive'])
    cm_df.to_csv('confusion_matrix.csv')
    print("Saved: confusion_matrix.csv")
    print("Saved: confusion_matrix.png")
else:
    print("Warning: Test set does not contain 'market_forward_excess_returns'. Skipping evaluation.")
    print("No evaluation CSVs generated.")