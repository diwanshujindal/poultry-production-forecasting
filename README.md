# Poultry Production Time Series Forecasting

This project implements an LSTM-based time series forecasting model to predict poultry production values across different Canadian regions.

## Dataset
The dataset contains Canadian poultry production statistics from 1976 to 2024, including various commodity types and geographical regions.

## Project Structure
- `data/`: Contains the raw dataset
- `notebooks/`: Jupyter notebooks for exploratory analysis
- `src/`: Python modules for data processing, modeling, and visualization
- `models/`: Saved trained models
- `results/`: Generated visualizations and results

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the main pipeline: `python main.py`

Or explore the notebook: `notebooks/exploratory_analysis.ipynb`

## Model
The project uses an LSTM neural network for multi-step time series forecasting with the following architecture:
- Two LSTM layers with 50 units each
- Dropout layers for regularization
- Dense output layer for multi-step predictions

## Results
The model achieves good performance in predicting poultry production values with appropriate validation and testing procedures.
