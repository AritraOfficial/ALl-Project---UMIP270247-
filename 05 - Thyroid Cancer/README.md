# Thyroid Cancer Recurrence Prediction System

This system predicts the likelihood of thyroid cancer recurrence in patients based on their clinical data. It uses a deep learning model to make predictions and provides a user-friendly interface for healthcare professionals.

## Features

- Data preprocessing and analysis of thyroid cancer patient data
- Deep learning model for predicting cancer recurrence with optimized threshold
- Techniques to handle class imbalance (SMOTE, class weighting)
- Visualizations to understand the data and model performance
- User-friendly Streamlit interface for making predictions

## Files in this Project

- `thyroid_cancer_model.ipynb`: Jupyter notebook for data analysis, preprocessing, and model training
- `app.py`: Streamlit application for the user interface
- `README.md`: This file with instructions

## How to Use

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow streamlit imbalanced-learn