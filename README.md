# Crop Recommendation System

A machine learning project for recommending suitable crops based on environmental and soil-related input features. The workflow is implemented in a notebook-style Python script and demonstrates data preprocessing, model training, evaluation, explainability, and prediction.

## Project Overview

This project uses a supervised classification approach to predict the most suitable crop from features such as:
- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

The implementation includes:
- Data loading and preprocessing
- Train/validation/test splitting
- Multiple classifier comparison
- Performance metrics such as accuracy, precision, recall, and F1-score
- Hyperparameter tuning for decision trees
- SHAP-based feature importance analysis
- LIME-based explanation support

## Files in This Repository

- [PycharmProjects/PythonProject/main.py](PycharmProjects/PythonProject/main.py) - Simple starter script
- [PycharmProjects/PythonProject/crop_recommendation_notebook.py](PycharmProjects/PythonProject/crop_recommendation_notebook.py) - Notebook-based experiment workflow
- [PycharmProjects/PythonProject/crop_recommendation_full_notebook.py](PycharmProjects/PythonProject/crop_recommendation_full_notebook.py) - Full workflow script covering the notebook cells and evaluation steps

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn shap seaborn lime
```

## How to Run

Run the full workflow script:

```bash
python PycharmProjects/PythonProject/crop_recommendation_full_notebook.py
```

Make sure the dataset files are available in the working directory:
- dataset.csv
- test.csv

## Model Workflow

1. Load the crop dataset
2. Encode the target labels
3. Split data into training, validation, and test sets
4. Train multiple classification models
5. Compare accuracy and other metrics
6. Tune the best-performing model
7. Generate explainability plots and reports

## Notes

- The project is intended for educational and demonstration purposes.
- The notebook-style implementation can be adapted for deployment into a web app or API later.
- SHAP and LIME may require additional environment support depending on your Python setup.

## Author

Reddyvandla Bhuvaneswar Reddy
