"""
Exploratory Data Analysis functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    """
    # Basic information
    print("Dataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Visualizations
    plot_missing_values(df)
    plot_distributions(df)
    plot_time_series(df, 'VALUE')
    
def plot_missing_values(df):
    """Plot heatmap of missing values"""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig('results/missing_values_heatmap.png')
    plt.close()

def plot_distributions(df):
    """Plot distributions of numerical features"""
    df.hist(figsize=(15, 10), bins=30, color='skyblue')
    plt.suptitle('Distribution of Features')
    plt.tight_layout()
    plt.savefig('results/feature_distributions.png')
    plt.close()

def plot_time_series(df, target_column):
    """Plot time series of target variable"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target_column])
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Change in Value over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/time_series_plot.png')
    plt.close()
