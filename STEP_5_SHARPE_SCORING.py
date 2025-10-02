import pandas as pd
import numpy as np

def ensure_required_columns(test_df):
    """
    Ensure test_df has forward_returns and risk_free_rate columns, using lagged versions if available,
    or generating synthetic data as a fallback for local testing.
    
    Args:
        test_df (pd.DataFrame): Test DataFrame, potentially containing lagged columns.
    
    Returns:
        pd.DataFrame: Modified DataFrame with required columns.
    """
    print("Checking for required columns in test_df...")
    test_df = test_df.copy()
    
    has_returns = 'forward_returns' in test_df.columns or 'lagged_forward_returns' in test_df.columns
    has_risk_free = 'risk_free_rate' in test_df.columns or 'lagged_risk_free_rate' in test_df.columns
    
    print(f"Has forward_returns or lagged_forward_returns: {has_returns}")
    print(f"Has risk_free_rate or lagged_risk_free_rate: {has_risk_free}")
    
    if not has_returns:
        print("Neither forward_returns nor lagged_forward_returns found. Generating synthetic forward_returns.")
        np.random.seed(42)
        test_df['forward_returns'] = np.random.normal(0.0005, 0.01, len(test_df))
    elif 'lagged_forward_returns' in test_df.columns and 'forward_returns' not in test_df.columns:
        print("Using lagged_forward_returns as forward_returns.")
        test_df['forward_returns'] = test_df['lagged_forward_returns']
    
    if not has_risk_free:
        print("Neither risk_free_rate nor lagged_risk_free_rate found. Generating synthetic risk_free_rate.")
        np.random.seed(42)
        test_df['risk_free_rate'] = np.full(len(test_df), 0.0001)
    elif 'lagged_risk_free_rate' in test_df.columns and 'risk_free_rate' not in test_df.columns:
        print("Using lagged_risk_free_rate as risk_free_rate.")
        test_df['risk_free_rate'] = test_df['lagged_risk_free_rate']
    
    print("Columns after processing:", list(test_df.columns))
    return test_df

def compute_sharpe_tester(df, vol_penalty_threshold=2.0, use_adjusted=True):
    """
    Computes the Hull Tactical competition metric: Modified Sharpe ratio for portfolio.
    
    Args:
    - df (pd.DataFrame): DataFrame with columns ['date_id', 'prediction', 'forward_returns', 'risk_free_rate', 'is_scored'].
    - vol_penalty_threshold (float): Multiple of market vol beyond which score is penalized (default 2.0).
    - use_adjusted (bool): If True, compute volatility-adjusted Sharpe.
    
    Returns:
    - dict: Scores and diagnostics.
    """
    print("\nStarting Sharpe ratio computation...")
    # Ensure required columns
    df = ensure_required_columns(df)
    
    # Filter to scored rows
    print(f"Filtering for scored rows (is_scored == True)...")
    scored_df = df[df['is_scored'] == True].copy()
    if scored_df.empty:
        print("No scored rows found. Returning NaN results.")
        return {'sharpe': np.nan, 'adjusted_sharpe': np.nan, 'vol_ratio': np.nan, 'mean_excess': np.nan, 'warning': 'No scored rows'}
    
    print(f"Number of scored rows: {len(scored_df)}")
    
    predictions = scored_df['prediction'].values
    forward_returns = scored_df['forward_returns'].values
    risk_free_rate = scored_df['risk_free_rate'].values
    
    print("Computing portfolio returns (prediction * forward_returns)...")
    port_returns = predictions * forward_returns
    print(f"Portfolio returns range: {np.min(port_returns):.6f} to {np.max(port_returns):.6f}")
    
    print("Computing excess portfolio returns...")
    port_excess = port_returns - risk_free_rate
    print(f"Excess returns mean: {np.mean(port_excess):.6f}")
    
    market_std = np.std(forward_returns)
    port_std = np.std(port_returns)
    print(f"Market volatility (std of forward_returns): {market_std:.6f}")
    print(f"Portfolio volatility (std of port_returns): {port_std:.6f}")
    
    if port_std == 0:
        print("Zero portfolio volatility detected. Returning neutral score.")
        return {'sharpe': 0.0, 'adjusted_sharpe': 0.0, 'vol_ratio': 0.0, 'mean_excess': np.mean(port_excess), 'warning': 'Zero portfolio volatility'}
    
    mean_excess = np.mean(port_excess)
    standard_sharpe = mean_excess / port_std
    print(f"Standard Sharpe ratio: {standard_sharpe:.6f}")
    
    vol_ratio = port_std / market_std if market_std > 0 else 0
    print(f"Volatility ratio (port_std / market_std): {vol_ratio:.6f}")
    penalized_sharpe = standard_sharpe if vol_ratio <= vol_penalty_threshold else 0.0
    print(f"Penalized Sharpe (after vol penalty, threshold={vol_penalty_threshold}): {penalized_sharpe:.6f}")
    
    if not use_adjusted:
        score = penalized_sharpe
        print("Using non-adjusted Sharpe as final score.")
    else:
        adjusted_sharpe = 0.0 if market_std == 0 else min(market_std / port_std, 1.0) * mean_excess / market_std
        score = adjusted_sharpe if vol_ratio <= vol_penalty_threshold else 0.0
        print(f"Adjusted Sharpe ratio: {adjusted_sharpe:.6f}")
        print("Using adjusted Sharpe as final score.")
    
    print("\nFinal Sharpe Tester Results:")
    return {
        'sharpe': standard_sharpe,
        'adjusted_sharpe': adjusted_sharpe if use_adjusted else penalized_sharpe,
        'score': score,
        'vol_ratio': vol_ratio,
        'mean_excess': mean_excess,
        'port_std': port_std,
        'market_std': market_std,
        'n_periods': len(scored_df)
    }

# Example usage with test.csv
if __name__ == "__main__":
    print("Loading test data from test.csv...")
    try:
        test_df = pd.read_csv('hull-tactical-market-prediction/test.csv')
        print(f"Test data loaded. Shape: {test_df.shape}")
        print("Test columns:", list(test_df.columns))
    except FileNotFoundError:
        print("Error: test.csv not found in hull-tactical-market-prediction directory.")
        exit(1)
    
    print("\nGenerating sample predictions...")
    submission = pd.DataFrame({
        'date_id': test_df['date_id'],
        'prediction': np.random.uniform(-0.01, 0.01, len(test_df))  # Replace with your model predictions
    })
    print(f"Sample submission created. Shape: {submission.shape}")
    
    print("\nMerging predictions with test data...")
    test_df = test_df.merge(submission, on='date_id')
    print(f"Merged data shape: {test_df.shape}")
    
    # Ensure required columns
    test_df = ensure_required_columns(test_df)
    
    # Run tester
    results = compute_sharpe_tester(test_df)
    for key, value in results.items():
        print(f"{key}: {value}")