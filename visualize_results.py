#!/usr/bin/env python3
"""Visualization script for training results using seaborn."""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def load_all_results(base_dir='./outputs'):
    """
    Load all CSV result files from the outputs directory.
    
    Searches for all CSV files in <base_dir>/**/results/*.csv
    
    Args:
        base_dir: Base directory to search for results
        
    Returns:
        Concatenated DataFrame with all results
    """
    csv_files = glob.glob(f'{base_dir}/**/results/*.csv', recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}/**/results/")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
    
    if not dfs:
        print("No CSV files could be loaded")
        return None
    
    results = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(results)} total records")
    
    return results


def plot_results_by_dropout_scatter(results_df, split='test', share_y=True, output_path=None):
    """
    Create a 2x2 subplot figure with seaborn scatterplots.
    
    X-axis: dropout_rate
    Y-axis: acc, balanced_acc, auroc, auc_pr
    
    Each point represents one seed run. Multiple points per dropout rate show the variance
    across different random seeds.
    
    Args:
        results_df: DataFrame with training results
        split: Which split to plot ('test', 'eval', 'train'). Default: 'test'
        share_y: Whether to use fixed y-axis limits for all subplots. Default: True
        output_path: Path to save the figure. If None, uses 'results_by_dropout_{split}.png'
    """
    
    # Filter for specified split
    split_results = results_df[results_df['split'] == split].copy()
    
    if len(split_results) == 0:
        print(f"No {split} results found")
        return
    
    # Get only the final epoch for each configuration (dropout_rate + seed)
    # Group by all identifier columns and take the last row (highest epoch)
    groupby_cols = ['model', 'dataset', 'seed', 'dropout_rate']
    available_cols = [col for col in groupby_cols if col in split_results.columns]
    
    final_results = split_results.sort_values('epoch').groupby(available_cols, as_index=False).last()
    
    print(f"Filtered to {len(final_results)} final epoch records from {len(split_results)} total {split} records")
    print(f"Unique dropout rates: {sorted(final_results['dropout_rate'].unique())}")
    print(f"Unique seeds: {sorted(final_results['seed'].unique())}")
    
    # Check if dropout_rate exists
    if 'dropout_rate' not in final_results.columns:
        print("dropout_rate not found in results")
        return
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"Effect of Dropout Rate on Model Performance ({split.capitalize()} Split)"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    metrics = ['acc', 'balanced_acc', 'auroc', 'auc_pr']
    axes_flat = axes.flatten()
    
    # Compute y-axis limits if share_y is True
    y_limits = {}
    if share_y:
        for metric in metrics:
            if metric in final_results.columns:
                y_min = final_results[metric].min()
                y_max = final_results[metric].max()
                y_limits[metric] = (y_min * 0.95, y_max * 1.05)  # Add 5% padding
    
    for idx, (ax, metric) in enumerate(zip(axes_flat, metrics)):
        if metric not in final_results.columns:
            ax.text(0.5, 0.5, f'{metric} not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric)
            continue
        
        # Create scatterplot with seaborn
        # Each point represents one seed run for a given dropout rate
        sns.scatterplot(
            data=final_results,
            x='dropout_rate',
            y=metric,
            ax=ax,
            s=100,
            alpha=0.7,
            palette='Set2',
            hue='seed' if 'seed' in final_results.columns else None,
            legend=None
            # legend=(idx == 0)  # Only show legend on first subplot
        )
        
        # Set fixed y-axis if enabled
        if share_y and metric in y_limits:
            ax.set_ylim(y_limits[metric])
        
        ax.set_xlabel('Dropout Rate', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    if output_path is None:
        output_path = f'results_by_dropout_scatter_{split}.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}\n")
    
    return fig, axes


def plot_results_by_dropout_box(results_df, split='test', share_y=True, output_path=None):
    """
    Create a 2x2 subplot figure with seaborn boxplots.
    
    X-axis: dropout_rate
    Y-axis: acc, balanced_acc, auroc, auc_pr
    
    Each box summarizes distribution across seeds at each dropout rate,
    using the final epoch value per (dropout_rate, seed).
    
    Args:
        results_df: DataFrame with training results
        split: Which split to plot ('test', 'eval', 'train'). Default: 'test'
        share_y: Whether to use fixed y-axis limits for all subplots. Default: True
        output_path: Path to save the figure. If None, uses 'results_box_by_dropout_{split}.png'
    """

    # Filter for specified split
    split_results = results_df[results_df['split'] == split].copy()
    if len(split_results) == 0:
        print(f"No {split} results found")
        return

    # Final epoch per configuration (model, dataset, seed, dropout_rate)
    groupby_cols = ['model', 'dataset', 'seed', 'dropout_rate']
    available_cols = [col for col in groupby_cols if col in split_results.columns]
    final_results = split_results.sort_values('epoch').groupby(available_cols, as_index=False).last()

    print(f"[BOX] Filtered to {len(final_results)} final epoch records from {len(split_results)} total {split} records")

    if 'dropout_rate' not in final_results.columns:
        print("dropout_rate not found in results")
        return

    # Figure and title (keep existing title style)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"Effect of Dropout Rate on Model Performance ({split.capitalize()} Split)"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    metrics = ['acc', 'balanced_acc', 'auroc', 'auc_pr']
    axes_flat = axes.flatten()

    # Shared y-axis limits if requested
    y_limits = {}
    if share_y:
        for metric in metrics:
            if metric in final_results.columns:
                y_min = final_results[metric].min()
                y_max = final_results[metric].max()
                y_limits[metric] = (y_min * 0.95, y_max * 1.05)

    for ax, metric in zip(axes_flat, metrics):
        if metric not in final_results.columns:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric)
            continue

        # Boxplot without legend (no hue)
        sns.boxplot(
            data=final_results,
            x='dropout_rate',
            y=metric,
            ax=ax,
            palette='Set2'
        )

        if share_y and metric in y_limits:
            ax.set_ylim(y_limits[metric])

        ax.set_xlabel('Dropout Rate', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    if output_path is None:
        output_path = f'results_box_by_dropout_box_{split}.png'

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved boxplot to: {output_path}\n")

    return fig, axes


def plot_train_results(results_df, share_y=True, output_path='results_by_dropout_train.png'):
    """Plot training results by dropout rate across all seeds."""
    return plot_results_by_dropout_scatter(results_df, split='train', share_y=share_y, output_path=output_path)


def plot_validation_results(results_df, share_y=True, output_path='results_by_dropout_validation.png'):
    """Plot validation/eval results by dropout rate across all seeds."""
    return plot_results_by_dropout_scatter(results_df, split='eval', share_y=share_y, output_path=output_path)


def plot_test_results(results_df, share_y=True, output_path='results_by_dropout_test.png'):
    """Plot test results by dropout rate across all seeds."""
    return plot_results_by_dropout_scatter(results_df, split='test', share_y=share_y, output_path=output_path)


def main():
    """Load results and generate visualizations for all splits."""
    
    # Load all CSV files
    results = load_all_results()
    
    if results is None or len(results) == 0:
        print("No results to visualize")
        return
    
    print(f"\nDataset overview:")
    print(f"  Unique models: {results['model'].unique()}")
    print(f"  Unique datasets: {results['dataset'].unique()}")
    print(f"  Unique splits: {results['split'].unique()}")
    
    if 'seed' in results.columns:
        print(f"  Unique seeds: {sorted(results['seed'].unique())}")
    if 'dropout_rate' in results.columns:
        print(f"  Unique dropout rates: {sorted(results['dropout_rate'].unique())}\n")
    
    # Generate plots for each split
    available_splits = results['split'].unique()
    
    for split in sorted(available_splits):
        print(f"Generating {split} plot...")
        plot_results_by_dropout_scatter(results, split=split, share_y=True)
        plot_results_by_dropout_box(results, split=split, share_y=True)


if __name__ == '__main__':
    main()
