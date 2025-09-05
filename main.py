"""
Main script to run the poultry production forecasting pipeline
"""

import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_data, preprocess_data, prepare_time_series_data
from src.eda import perform_eda
from src.model import build_model, train_model, evaluate_model, save_model
from src.utils import setup_directories, load_config
import json

def main():
    """Main pipeline function"""
    print("Setting up directories...")
    setup_directories()
    
    # Load configuration
    config = load_config()
    
    print("Loading and preprocessing data...")
    df = load_data('data/32100122.csv')
    df_processed = preprocess_data(df)
    
    print("Performing exploratory data analysis...")
    perform_eda(df_processed)
    
    print("Preparing time series data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_x, scaler_y = prepare_time_series_data(
        df_processed, 
        target_column='VALUE',
        input_size=config['input_size'],
        output_size=config['output_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    print("Building and training model...")
    model, history = build_and_train_model(
        X_train, y_train, 
        X_val, y_val,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_size=config['output_size'],
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    print("Evaluating model...")
    results = evaluate_model(model, history, scaler_y, X_test, y_test)
    
    # Save model and results
    print("Saving model and results...")
    save_model(model, 'models/lstm_poultry_model.h5')
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Pipeline completed successfully!")
    print(f"Final test loss: {results['test_loss']:.4f}")

def build_and_train_model(X_train, y_train, X_val, y_val, input_shape, output_size, epochs=20, batch_size=32):
    """Build and train the model"""
    model = build_model(input_shape, output_size)
    model, history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)
    return model, history

if __name__ == "__main__":
    main()
