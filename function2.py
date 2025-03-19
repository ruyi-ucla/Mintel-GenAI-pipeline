import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def stat_summary(df):
    """
    Generates a structured statistical summary for the given dataframe
    Returns a dictionary with numerical summaries and visualization paths
    """
    # Define columns to analyze
    columns = ['name', 'region', 'supercat', 'cat', 'sub',
               'f_cat', 'year_month', 'year', 'month', 'flavor_prevalence']
    
    summary_data = {}  # Stores numerical summaries
    visualization_paths = []  # Stores paths to generated plots
    
    for col in columns:
        if col not in df.columns:
            print(f"Wanring: Column {col} not found in DataFrame")
            continue  # Skip missing columns

        unique_vals = df[col].nunique()

        # Store unique values if there's only one unique value
        if unique_vals == 1:
            summary_data[col] = {"unique_value": df[col].unique()[0]}
            continue  # No need to visualize single-value columns

        # For numerical columns, compute descriptive statistics
        if is_numeric_dtype(df[col]):
            summary_data[col] = df[col].describe().to_dict()

            # Generate density plot
            plt.figure(figsize=(6, 4))
            df[col].dropna().plot(kind='density', title=f"Density Plot of {col}")
            plt.xlabel(col)
            img_path = f"visual_{col}.png"
            plt.savefig(img_path)
            plt.close()
            visualization_paths.append(img_path)

        # For categorical columns, compute value counts
        else:
            value_counts = df[col].value_counts().to_dict()
            summary_data[col] = value_counts

            # Generate bar chart
            plt.figure(figsize=(8, 4))
            pd.Series(value_counts).plot(kind='bar', title=f"Bar Chart of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            img_path = f"visual_{col}.png"
            plt.savefig(img_path)
            plt.close()
            visualization_paths.append(img_path)

    # Return structured summary + visualization paths
    return {
        "summary": summary_data,
        "visualization_paths": visualization_paths
    }


def flavor_state(df):
    return None

def generate_plot(df):
    return None