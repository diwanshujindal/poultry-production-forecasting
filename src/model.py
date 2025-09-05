"""
Model building, training and evaluation functions
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape, output_size):
    """
    Build LSTM model architecture
    
    Parameters:
    input_shape (tuple): Shape of input data
    output_size (int): Size of output horizon
    
    Returns:
    tensorflow.keras.Model: Compiled model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=output_size)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """
    Train the LSTM model
    
    Parameters:
    model: Compiled model
    X_train: Training features
    y_train: Training targets
    X_val: Validation features
    y_val: Validation targets
    epochs: Number of training epochs
    batch_size: Batch size for training
    
    Returns:
    tuple: (trained_model, training_history)
    """
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history

def evaluate_model(model, history, scaler, X_test, y_test):
    """
    Evaluate model performance and create visualizations
    
    Parameters:
    model: Trained model
    history: Training history
    scaler: Fitted scaler for inverse transformation
    X_test: Test features
    y_test: Test targets
    """
    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('LSTM Model Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_history.png')
    plt.close()
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Plot predictions for first horizon
    y_pred_horizon0 = y_pred[:, 0]
    y_test_horizon0 = y_test_original[:, 0]
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_horizon0, label="Real Values", linestyle="-", marker="o")
    plt.plot(y_pred_horizon0, label="Predicted Values", linestyle="-", marker="x")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Real vs. Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/predictions_vs_actual.png')
    plt.close()
    
    # Calculate evaluation metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
