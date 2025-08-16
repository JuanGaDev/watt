import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_datasets(original_path='dataset.csv', synthetic_path='synthetic_data_ctgan.csv'):
    """Load both datasets"""
    original = pd.read_csv(original_path)
    synthetic = pd.read_csv(synthetic_path)
    return original, synthetic

def clean_data(df):
    """Clean the dataset by handling missing values and converting types"""
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Convert numeric columns
    numeric_columns = ['Estrato', 'Acueducto_m3', 'Acueducto_Valor', 'Alcantarillado_m3', 
                      'Alcantarillado_Valor', 'Energia_kWh', 'Energia_Valor', 'Gas_m3', 'Gas_Valor']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def plot_correlation_comparison(original, synthetic, figsize=(15, 6)):
    """Plot correlation matrices for both datasets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Select only numeric columns
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    orig_numeric = original[numeric_cols].dropna()
    synth_numeric = synthetic[numeric_cols].dropna()
    
    # Original correlation
    orig_corr = orig_numeric.corr()
    im1 = ax1.imshow(orig_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Original Dataset - Correlation Matrix')
    ax1.set_xticks(range(len(numeric_cols)))
    ax1.set_yticks(range(len(numeric_cols)))
    ax1.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax1.set_yticklabels(numeric_cols)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = ax1.text(j, i, f'{orig_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # Synthetic correlation
    synth_corr = synth_numeric.corr()
    im2 = ax2.imshow(synth_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax2.set_title('Synthetic Dataset - Correlation Matrix')
    ax2.set_xticks(range(len(numeric_cols)))
    ax2.set_yticks(range(len(numeric_cols)))
    ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax2.set_yticklabels(numeric_cols)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = ax2.text(j, i, f'{synth_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    return fig

def plot_trend_comparison(original, synthetic, figsize=(15, 10)):
    """Plot trend comparison using line charts and Gaussian bell curves"""
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    
    # Calculate number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx]
        
        # Remove NaN values
        orig_data = original[col].dropna()
        synth_data = synthetic[col].dropna()
        
        if len(orig_data) > 0 and len(synth_data) > 0:
            # Create histogram data for line chart
            orig_hist, orig_bins = np.histogram(orig_data, bins=min(10, len(orig_data)), density=True)
            synth_hist, synth_bins = np.histogram(synth_data, bins=min(10, len(synth_data)), density=True)
            
            # Calculate bin centers for line chart
            orig_bin_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
            synth_bin_centers = (synth_bins[:-1] + synth_bins[1:]) / 2
            
            # Plot line chart
            ax.plot(orig_bin_centers, orig_hist, 'o-', label='Original', color='skyblue', linewidth=2, markersize=6)
            ax.plot(synth_bin_centers, synth_hist, 's-', label='Synthetic', color='lightcoral', linewidth=2, markersize=6)
            
            # Fit Gaussian curves
            try:
                from scipy.stats import norm
                
                # Fit normal distribution to original data
                orig_mean, orig_std = norm.fit(orig_data)
                orig_x = np.linspace(orig_data.min(), orig_data.max(), 100)
                orig_gaussian = norm.pdf(orig_x, orig_mean, orig_std)
                
                # Fit normal distribution to synthetic data
                synth_mean, synth_std = norm.fit(synth_data)
                synth_x = np.linspace(synth_data.min(), synth_data.max(), 100)
                synth_gaussian = norm.pdf(synth_x, synth_mean, synth_std)
                
                # Plot Gaussian curves
                ax.plot(orig_x, orig_gaussian, '--', color='skyblue', alpha=0.7, linewidth=2, label='Original Gaussian')
                ax.plot(synth_x, synth_gaussian, '--', color='lightcoral', alpha=0.7, linewidth=2, label='Synthetic Gaussian')
                
            except ImportError:
                pass  # scipy not available, skip Gaussian fitting
            
            ax.set_title(f'Trend Comparison - {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_gaussian_bell_comparison(original, synthetic, figsize=(15, 10)):
    """Plot Gaussian bell curve comparison for all numerical variables"""
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    
    # Calculate number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx]
        
        # Remove NaN values
        orig_data = original[col].dropna()
        synth_data = synthetic[col].dropna()
        
        if len(orig_data) > 0 and len(synth_data) > 0:
            try:
                from scipy.stats import norm
                
                # Create x-axis range
                min_val = min(orig_data.min(), synth_data.min())
                max_val = max(orig_data.max(), synth_data.max())
                x = np.linspace(min_val, max_val, 200)
                
                # Fit and plot original data Gaussian
                orig_mean, orig_std = norm.fit(orig_data)
                orig_gaussian = norm.pdf(x, orig_mean, orig_std)
                ax.plot(x, orig_gaussian, color='skyblue', linewidth=3, label=f'Original (μ={orig_mean:.2f}, σ={orig_std:.2f})')
                
                # Fill area under original curve
                ax.fill_between(x, orig_gaussian, alpha=0.3, color='skyblue')
                
                # Fit and plot synthetic data Gaussian
                synth_mean, synth_std = norm.fit(synth_data)
                synth_gaussian = norm.pdf(x, synth_mean, synth_std)
                ax.plot(x, synth_gaussian, color='lightcoral', linewidth=3, label=f'Synthetic (μ={synth_mean:.2f}, σ={synth_std:.2f})')
                
                # Fill area under synthetic curve
                ax.fill_between(x, synth_gaussian, alpha=0.3, color='lightcoral')
                
                # Add vertical lines for means
                ax.axvline(orig_mean, color='navy', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvline(synth_mean, color='darkred', linestyle='--', alpha=0.7, linewidth=2)
                
                ax.set_title(f'Gaussian Bell Comparison - {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Probability Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except ImportError:
                # Fallback if scipy is not available
                ax.text(0.5, 0.5, 'scipy required for Gaussian fitting', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Gaussian Bell Comparison - {col}')
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate all comparison plots"""
    print("Loading datasets...")
    original, synthetic = load_datasets()
    
    print("Cleaning data...")
    original_clean = clean_data(original)
    synthetic_clean = clean_data(synthetic)
    
    print("Generating plots...")
    
    # Create output directory
    output_dir = Path("distribution_plots")
    output_dir.mkdir(exist_ok=True)
    
    # 2. Numerical variables distributions
    numeric_columns = ['Acueducto_m3', 'Acueducto_Valor', 'Alcantarillado_m3', 
                      'Alcantarillado_Valor', 'Energia_kWh', 'Energia_Valor', 'Gas_m3', 'Gas_Valor']
    
    # 3. Correlation comparison
    print("Plotting correlation matrices...")
    fig = plot_correlation_comparison(original_clean, synthetic_clean)
    fig.savefig(output_dir / 'correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Trend comparison with line charts and Gaussian curves
    print("Plotting trend comparisons...")
    fig = plot_trend_comparison(original_clean, synthetic_clean)
    fig.savefig(output_dir / 'trend_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Gaussian bell curve comparison
    print("Plotting Gaussian bell curves...")
    fig = plot_gaussian_bell_comparison(original_clean, synthetic_clean)
    fig.savefig(output_dir / 'gaussian_bell_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"All plots saved in '{output_dir}' directory!")
    print("Generated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")

def generate_comparison_report(original, synthetic, output_dir):
    """Generate a text report with key statistics"""
    report_path = output_dir / "comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DATASET COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASET SIZES:\n")
        f.write(f"Original dataset: {len(original)} rows, {len(original.columns)} columns\n")
        f.write(f"Synthetic dataset: {len(synthetic)} rows, {len(synthetic.columns)} columns\n\n")
        
        f.write("MISSING VALUES:\n")
        f.write("Original dataset:\n")
        f.write(original.isnull().sum().to_string())
        f.write("\n\nSynthetic dataset:\n")
        f.write(synthetic.isnull().sum().to_string())
        f.write("\n\n")
        
        f.write("NUMERICAL STATISTICS COMPARISON:\n")
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            f.write(f"\n{col}:\n")
            f.write(f"  Original - Mean: {original[col].mean():.2f}, Std: {original[col].std():.2f}\n")
            f.write(f"  Synthetic - Mean: {synthetic[col].mean():.2f}, Std: {synthetic[col].std():.2f}\n")
        
        f.write("\n\nCATEGORICAL VARIABLES:\n")
        categorical_cols = original.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            f.write(f"\n{col}:\n")
            f.write("  Original distribution:\n")
            f.write(original[col].value_counts().to_string())
            f.write("\n  Synthetic distribution:\n")
            f.write(synthetic[col].value_counts().to_string())
            f.write("\n")

if __name__ == "__main__":
    main()
