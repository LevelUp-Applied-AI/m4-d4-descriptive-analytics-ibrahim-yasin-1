"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    """Compute summary statistics for all numeric columns.

    Args:
        df: pandas DataFrame with at least some numeric columns

    Returns:
        DataFrame containing count, mean, median, std, min, max
        for each numeric column. Saves the result to output/summary.csv.
    """
    numeric_df = df.select_dtypes(include="number")

    summary = pd.DataFrame(
        [
            numeric_df.count(),
            numeric_df.mean(),
            numeric_df.median(),
            numeric_df.std(),
            numeric_df.min(),
            numeric_df.max(),
        ],
        index=["count", "mean", "median", "std", "min", "max"],
    )

    os.makedirs("output", exist_ok=True)
    summary.to_csv("output/summary.csv")

    return summary


def plot_distributions(df, columns, output_path):
    """Create a 2x2 subplot figure with histograms for the specified columns.

    Args:
        df: pandas DataFrame
        columns: list of 4 column names to plot
        output_path: file path to save the figure

    Returns:
        None — saves the figure to output_path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation(df, output_path):
    """Compute Pearson correlation matrix and visualize as a heatmap.

    Args:
        df: pandas DataFrame with numeric columns
        output_path: file path to save the figure

    Returns:
        None — saves the figure to output_path
    """
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr(method="pearson")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Load data, compute summary, and generate all plots."""
    os.makedirs("output", exist_ok=True)

    data_path = "data/sample_sales.csv"
    df = pd.read_csv(data_path)

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["sales"] = df["quantity"] * df["unit_price"]

    compute_summary(df)

    plot_distributions(
        df,
        ["quantity", "unit_price", "month", "sales"],
        "output/distributions.png",
    )

    plot_correlation(df, "output/correlation.png")


if __name__ == "__main__":
    main()