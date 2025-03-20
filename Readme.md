*# Credit Card Fraud Detection System

A machine learning-based system for detecting fraudulent credit card transactions with high accuracy and minimal false positives.

## Project Overview

This project implements a comprehensive fraud detection system that:

1. Processes and analyzes credit card transaction data
2. Handles class imbalance using SMOTE
3. Trains and evaluates multiple machine learning models
4. Identifies the most important features for fraud detection
5. Provides visualization tools for model performance analysis

## System Performance

Based on the latest execution:

- **Dataset size**: 1,296,675 transactions
- **Fraud cases**: 7,506 (0.58% of total transactions)
- **Models trained**: Random Forest, Gradient Boosting
- **Best model**: Random Forest (F1-Score: 0.6480)
- **Key performance metrics**:
    - Random Forest: 79% recall on fraud cases with 55% precision
    - Gradient Boosting: 86% recall on fraud cases with 18% precision

## Repository Structure

```
fraud-detection/
├── main.py              # Main script to run the fraud detection pipeline
├── utils.py             # Utility functions for data processing and visualization
├── requirements.txt     # Project dependencies
├── fraudTrain.rar      # Training dataset
├── models/              # Saved models and preprocessing objects
│   ├── random_forest_20250320_205116.pkl
│   ├── gradient_boosting_20250320_205116.pkl
│   └── preprocessing_20250320_205116.pkl
└── plots/               # Generated visualizations
        ├── fraud_distribution.png
        ├── feature_importance_random_forest.png
        └── feature_importance_gradient_boosting.png
```

## Features

### Data Processing

- **Datetime processing**: Extracts hour, day, month, and day of week from transaction timestamps
- **Geographic analysis**: Calculates distances between customer and merchant locations
- **Categorical encoding**: Handles merchant, category, job, and gender features
- **Feature scaling**: Normalizes numerical features

### Model Training

- **Random Forest**: Optimized for balanced precision and recall
- **Gradient Boosting**: Provides high recall for fraud detection
- **Class imbalance handling**: Uses SMOTE to address the imbalanced nature of fraud data (0.58% fraud)

### Model Evaluation

- **Comprehensive metrics**: F1-score, precision, recall, and confusion matrix
- **Feature importance analysis**: Identifies the most important features for fraud detection
- **Cross-validation**: Ensures model reliability and generalization

## Key Findings

### Most Important Features for Fraud Detection

1. **Transaction amount** (amt): By far the most significant indicator of fraud
2. **Transaction category**: Different categories have varying fraud risks
3. **Hour of transaction**: Time of day significantly impacts fraud likelihood
4. **Population density** (city_pop): Transactions in certain population areas show higher fraud rates
5. **Merchant**: Specific merchants may have higher fraud rates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Create and activate a virtual environment:
     
     For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

     For Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Fraud Detection Pipeline

To run the full fraud detection pipeline:

```bash
python main.py
```

This will:
- Load and preprocess the data from fraudTrain.csv
- Handle class imbalance with SMOTE
- Train Random Forest and Gradient Boosting models
- Evaluate model performance
- Save the trained models to the models/ directory
- Generate visualizations in the plots/ directory

### Making Predictions

To use the trained models for prediction with your own script:

```python
import pickle
import pandas as pd
from utils import preprocess_data

# Load the model and preprocessing objects
with open('models/random_forest_20250320_205116.pkl', 'rb') as f:
        model = pickle.load(f)
        
with open('models/preprocessing_20250320_205116.pkl', 'rb') as f:
        preprocessing = pickle.load(f)

# First, unzip the fraudTrain.rar file
# You need to have appropriate software like WinRAR, 7-Zip, or unrar installed
import os
import subprocess

# For Windows using 7-Zip (adjust path if needed)
if os.name == 'nt':
    subprocess.run(['7z', 'x', 'fraudTrain.rar'])
# For Linux/Mac using unrar
else:
    subprocess.run(['unrar', 'x', 'fraudTrain.rar'])

# Load and preprocess new transaction data
new_data = pd.read_csv('fraudTrain.csv')
preprocessed_data = preprocess_data(new_data, preprocessing)

# Make predictions
predictions = model.predict(preprocessed_data)
```

## Performance Optimization

The system is designed to handle large transaction volumes efficiently. The pipeline includes:

- Optimized preprocessing steps
- Efficient model training with parallel processing (Random Forest uses n_jobs=-1)
- Fast prediction capabilities for real-time fraud detection

## Future Improvements

- Implement deep learning models for improved performance
- Add anomaly detection techniques for identifying new fraud patterns
- Develop real-time monitoring dashboard
- Implement model explainability features for better interpretability
- Create an API for real-time fraud detection

## Dataset

The system was trained using the fraudTrain.csv dataset and evaluated on fraudTest.csv, which contain credit card transaction data with the following features:

- Transaction details (date, time, amount)
- Credit card information
- Merchant information
- Customer demographics
- Geographic coordinates
- Fraud labels (0.58% of transactions labeled as fraud)*