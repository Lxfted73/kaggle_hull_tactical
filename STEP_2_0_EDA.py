import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import math

# Load the train dataset (adjust path if running locally)
train_path = 'hull-tactical-market-prediction/train.csv'
train = pd.read_csv(train_path)

# Target variable
target_col = 'market_forward_excess_returns'

# Features
feature_cols = [col for col in train.columns if col != target_col]
print(f"Dataset shape: {train.shape}")
print(f"Target: {target_col}")
print(f"Number of features: {len(feature_cols)}")

# Null counts
null_counts = train.isnull().sum()
null_df = pd.DataFrame({'Column': null_counts.index, 'Null_Count': null_counts.values})
non_zero_nulls = null_df[null_df['Null_Count'] > 0]

# Select numeric features for histograms
numeric_features = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
n_features = len(numeric_features)

# Correlation for heatmap
print("\n=== Correlation Analysis ===")
corrs = train[feature_cols].corrwith(train[target_col]).abs().sort_values(ascending=False)
top_n = 20
top_features = corrs.head(top_n).index.tolist()
corr_df = train[top_features + [target_col]].corr()

# Save all plots to PDF
pdf_path = 'eda_analysis.pdf'
with PdfPages(pdf_path) as pdf:
    
    # Page 1: Dataset Head
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    head_df = train.head().round(3)  # First 5 rows, rounded for readability
    table_data = [head_df.columns.tolist()] + head_df.values.tolist()
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)  # Adjust table size
    ax.set_title('Dataset Head (First 5 Rows)', fontsize=12)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Page 2: Summary Statistics
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    summary = train.describe().round(3).T  # Summary stats (mean, std, min, max, etc.)
    summary_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    table_data = summary[summary_cols].values.tolist()
    table = ax.table(
        cellText=table_data,
        colLabels=summary_cols,
        rowLabels=summary.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Page 3: Null Counts Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(non_zero_nulls) > 0:
        sns.barplot(data=non_zero_nulls, x='Column', y='Null_Count', ax=ax)
        ax.set_title('Null Counts per Column')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No Null Values Found', ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Null Counts')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.2)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Pages for Histograms (up to 10 features per page, 5x2 grid)
    features_per_page = 10
    nrows = 5
    ncols = 2
    figsize = (12, 10)
    bins = 30
    
    n_pages = math.ceil(n_features / features_per_page)
    
    for page in range(n_pages):
        # Select features for this page
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, n_features)
        page_features = numeric_features[start_idx:end_idx]
        
        # Create figure and subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()  # Flatten for easier iteration
        
        # Plot each feature
        for i, feature in enumerate(page_features):
            axes[i].hist(train[feature], bins=bins, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{feature}', fontsize=10)
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)
            axes[i].tick_params(axis='both', labelsize=7)
        
        # Turn off unused axes
        for i in range(len(page_features), len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.3, hspace=0.5)
        
        # Add super title
        fig.suptitle(f'Feature Distributions (Page {page + 1})', fontsize=12, y=0.98)
        
        # Save to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # Last Page: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=False, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title(f'Correlation Heatmap: Top {top_n} Features + Target')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"\nEDA plots saved to {pdf_path}")
print("Top 10 absolute correlations with target:")
print(corrs.head(10))