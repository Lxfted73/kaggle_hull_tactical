import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

# Load the train dataset (adjust path if running locally)
train_path = 'train_cleaned.csv'
train = pd.read_csv(train_path)

# Target variable
target_col = 'market_forward_excess_returns'

# Features
feature_cols = [col for col in train.columns if col != target_col]
print(f"Dataset shape: {train.shape}")
print(f"Target: {target_col}")
print(f"Number of features: {len(feature_cols)}")

# Null counts
print("\n=== Null Counts ===")
null_counts = train.isnull().sum()
null_df = pd.DataFrame({'Column': null_counts.index, 'Null_Count': null_counts.values})
non_zero_nulls = null_df[null_df['Null_Count'] > 0]

# Small histograms for features (multiple per page, e.g., 10 per figure)
numeric_features = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
n_features = len(numeric_features)

# Print summary statistics for numeric features
print("\n=== Summary Statistics for Numeric Features ===")
print(train[numeric_features].describe())

# Correlation for heatmap
print("\n=== Correlation Analysis ===")
corrs = train[feature_cols].corrwith(train[target_col]).abs().sort_values(ascending=False)
top_n = 20
top_features = corrs.head(top_n).index.tolist()
corr_df = train[top_features + [target_col]].corr()

# Save all plots to PDF
pdf_path = 'eda_analysis.pdf'
with PdfPages(pdf_path) as pdf:
    
    # Page 1: Null Counts Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(non_zero_nulls) > 0:
        sns.barplot(data=non_zero_nulls, x='Column', y='Null_Count', ax=ax)
        ax.set_title('Null Counts per Column')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No Null Values Found', ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Null Counts')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    # Pages for Summary Statistics (10 features per page, starting from Page 2)
    summary_df = train[numeric_features].describe().round(4)  # Round for readability
    summary_rows = summary_df.index.tolist()
    stats_per_page = 10
    n_summary_pages = (n_features + stats_per_page - 1) // stats_per_page
    current_page = 2  # Starting after nulls
    
    for page in range(n_summary_pages):
        start_idx = page * stats_per_page
        end_idx = min(start_idx + stats_per_page, n_features)
        page_features = numeric_features[start_idx:end_idx]
        page_summary_df = summary_df[page_features]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        summary_data = page_summary_df.values
        summary_cols = page_summary_df.columns.tolist()
        table = ax.table(cellText=summary_data,
                         colLabels=summary_cols,
                         rowLabels=summary_rows,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)  # Adjust for multi-row
        ax.set_title(f'Summary Statistics for Numeric Features - Features {start_idx+1} to {end_idx} (Page {current_page})', fontsize=14, pad=20)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        current_page += 1
    
    # Pages for Histograms (10 per page)
    figs_per_page = 10
    n_hist_pages = (n_features + figs_per_page - 1) // figs_per_page
    
    for page in range(n_hist_pages):
        start_idx = page * figs_per_page
        end_idx = min(start_idx + figs_per_page, n_features)
        page_features = numeric_features[start_idx:end_idx]
        
        n_plots = len(page_features)
        cols = 5
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.ravel() if rows > 1 or cols > 1 else [axes]
        
        for i, col in enumerate(page_features):
            if i < n_plots:
                train[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(col, fontsize=8)
                axes[i].tick_params(axis='x', labelsize=6)
                axes[i].tick_params(axis='y', labelsize=6)
            else:
                axes[i].axis('off')
        
        plt.suptitle(f'Histograms of Features - Features {start_idx+1} to {end_idx} (Page {current_page})', fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        current_page += 1
    
    # Last Page: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=False, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title(f'Correlation Heatmap: Top {top_n} Features + Target (Page {current_page})')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"\nEDA plots saved to {pdf_path}")
print("Top 10 absolute correlations with target:")
print(corrs.head(10))