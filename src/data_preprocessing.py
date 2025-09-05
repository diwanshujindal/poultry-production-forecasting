"""
Data loading and preprocessing functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
    
    # Handle remaining missing values
    df_processed = df_processed.dropna()
    
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

def prepare_time_series_data(df, target_column, input_size=7, output_size=2, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare time series data for modeling
    
    Parameters:
    df (pandas.DataFrame): Processed data
    target_column (str): Name of target column
    input_size (int): Size of input window
    output_size (int): Size of output horizon
    train_ratio (float): Proportion of data for training
    val_ratio (float): Proportion of data for validation
    
    Returns:
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler_x, scaler_y)
    """
    # Create sequences
    X, y = create_window_sequences(df, target_column, input_size, output_size)
    
    # Split data
    train_idx = int(len(X) * train_ratio)
    val_idx = int(len(X) * (train_ratio + val_ratio))
    
    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
    
    # Scale features
    scaler_x = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler_x.fit(X_train_reshaped)
    
    X_train_scaled = scaler_x.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_x.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Scale targets
    scaler_y = MinMaxScaler()
    y_train_reshaped = y_train.reshape(-1, 1)
    scaler_y.fit(y_train_reshaped)
    
    y_train_scaled = scaler_y.transform(y_train_reshaped).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    print(f"Training set: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"Validation set: X={X_val_scaled.shape}, y={y_val_scaled.shape}")
    print(f"Test set: X={X_test_scaled.shape}, y={y_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_x, scaler_y
