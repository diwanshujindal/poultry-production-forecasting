"""
Data loading and preprocessing functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Load the dataset from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, 
    removing unnecessary columns, and feature engineering
    
    Parameters:
    df (pandas.DataFrame): Raw dataset
    
    Returns:
    pandas.DataFrame: Preprocessed dataset
    """
    # Convert REF_DATE to datetime
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'], format='%Y-%m')
    
    # Remove columns with more than 80% missing values
    threshold = 0.8
    dropped_columns = []
    for column in df.columns:
        missing_percentage = df[column].isnull().sum() / len(df)
        if missing_percentage > threshold:
            dropped_columns.append(column)
            df = df.drop(column, axis=1)
    
    print(f"Dropped columns with >80% missing values: {dropped_columns}")
    
    # Remove columns with only one unique value
    removed_columns = df.columns[df.nunique() == 1].tolist()
    df.drop(columns=removed_columns, inplace=True)
    print(f"Dropped columns with single unique value: {removed_columns}")
    
    # Remove redundant columns
    if 'DGUID' in df.columns:
        df = df.drop('DGUID', axis=1)
        print("Dropped redundant DGUID column")
    
    # Filter for Canada and specific commodity
    df_canada = df[df['GEO'] == 'Canada']
    df_commodity = df_canada[df_canada['Commodity'] == 'All poultry meat, total']
    df_processed = df_commodity[['REF_DATE', 'VALUE']].set_index('REF_DATE')
    
    # Extract datetime features
    df_processed = extract_datetime_features(df_processed)
    
    return df_processed

def extract_datetime_features(df):
    """
    Extract time-based features from datetime index
    
    Parameters:
    df (pandas.DataFrame): DataFrame with datetime index
    
    Returns:
    pandas.DataFrame: DataFrame with added time features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Season'] = df.index.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    df['Days_in_Month'] = df.index.days_in_month
    
    # Create season dummies
    season_dummies = pd.get_dummies(df['Season'], prefix='Season', drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)
    df = df.drop('Season', axis=1)
    
    return df

def create_window_sequences(df, target_column, input_size=7, output_size=2):
    """
    Create window sequences for time series forecasting
    
    Parameters:
    df (pandas.DataFrame): Input data
    target_column (str): Name of target column
    input_size (int): Size of input window
    output_size (int): Size of output horizon
    
    Returns:
    tuple: (X, y) arrays for model training
    """
    X, y = [], []
    features = df.columns.tolist()
    
    for i in range(len(df) - input_size - output_size + 1):
        X_window = df[features].values[i : i + input_size]
        y_window = df[target_column].values[i + input_size : i + input_size + output_size]
        
        X.append(X_window)
        y.append(y_window)
    
    return np.array(X), np.array(y)
