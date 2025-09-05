"""
Model building, training and evaluation functions
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

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
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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

def evaluate_model(model, history, scaler_y, X_test, y_test):
    """
    Evaluate model performance and create visualizations
    
    Parameters:
    model: Trained model
    history: Training history
    scaler_y: Fitted scaler for target variable
    X_test: Test features
    y_test: Test targets
    
    Returns:
    dict: Evaluation results
    """
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Calculate evaluation metrics for each horizon
    metrics = {}
    for horizon in range(y_test.shape[1]):
        y_test_h = y_test_original[:, horizon]
        y_pred_h = y_pred[:, horizon]
        
        metrics[f'horizon_{horizon}'] = {
            'mse': mean_squared_error(y_test_h, y_pred_h),
            'mae': mean_absolute_error(y_test_h, y_pred_h),
            'rmse': np.sqrt(mean_squared_error(y_test_h, y_pred_h)),
            'r2': r2_score(y_test_h, y_pred_h)
        }
    
    # Overall metrics
    overall_metrics = {
        'mse': mean_squared_error(y_test_original.flatten(), y_pred.flatten()),
        'mae': mean_absolute_error(y_test_original.flatten(), y_pred.flatten()),
        'rmse': np.sqrt(mean_squared_error(y_test_original.flatten(), y_pred.flatten())),
        'r2': r2_score(y_test_original.flatten(), y_pred.flatten())
    }
    
    # Create visualizations
    plot_training_history(history)
    plot_predictions(y_test_original, y_pred)
    
    # Compile results
    results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'horizon_metrics': metrics,
        'overall_metrics': overall_metrics,
        'model_summary': model_summary_to_dict(model)
    }
    
    return results

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Model MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_test, y_pred):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(15, 10))
    
    # Plot for each horizon
    for horizon in range(y_test.shape[1]):
        plt.subplot(2, 2, horizon + 1)
        plt.plot(y_test[:, horizon], label="Actual", linestyle="-", marker="o", markersize=3)
        plt.plot(y_pred[:, horizon], label="Predicted", linestyle="-", marker="x", markersize=3)
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.title(f"Horizon {horizon + 1}: Actual vs Predicted")
        plt.legend()
        plt.grid(True)
    
    # Overall comparison
    plt.subplot(2, 2, 4)
    plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (All Horizons)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

def model_summary_to_dict(model):
    """Convert model summary to dictionary"""
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return {
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape),
                'param_count': layer.count_params()
            }
            for layer in model.layers
        ],
        'total_params': model.count_params(),
        'text_summary': '\n'.join(summary)
    }

def save_model(model, filepath):
    """
    Save trained model to file
    
    Parameters:
    model: Trained model
    filepath: Path to save the model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_saved_model(filepath):
    """
    Load a saved model from file
    
    Parameters:
    filepath: Path to the saved model
    
    Returns:
    tensorflow.keras.Model: Loaded model
    """
    if os.path.exists(filepath):
        return load_model(filepath)
    else:
        raise FileNotFoundError(f"Model file not found: {filepath}")
