
# Blood Donation Forecast

This repository contains a machine learning pipeline for predicting blood donation behavior using the TPOTClassifier for automated machine learning (AutoML) and LogisticRegression for a baseline classification model. The dataset used is the Blood Donation dataset, which provides features related to past blood donations and other attributes.

## Overview
The project demonstrates how to preprocess data, build predictive models, and evaluate their performance. The following steps are included:


1. ### Loading and Preparing Data:
- This line loads the Blood Donation dataset from a CSV file into a pandas DataFrame called df.

2. ### Data Preprocessing
- The column name whether he/she donated blood in March 2007 is renamed to Blood donated for easier reference.
- MinMaxScaler scales the Monetary (c.c. blood) column to a range between 0 and 1.
- A new column Monetary is created with the scaled values.
- The original Monetary (c.c. blood) column is then replaced with the scaled values, and the temporary Monetary column is dropped.

3. ### Feature and Target Variable Separation
- X contains all columns except the last one (Blood donated), which represents the features.
- y contains the Blood donated column, which is the target variable to be predicted.

4. ### Data Splitting
- The data is split into training and testing sets.
- test_size=0.2 means 20% of the data is used for testing, and 80% is used for training.
- random_state=42 ensures reproducibility of the split.

5. ### Model Training and Evaluation
- #### Automated Machine Learning with TPOT
  1. generations=8: Number of iterations for the genetic algorithm.
  2. population_size=20: Number of pipelines in each generation.
  3. verbosity=2: Level of output verbosity.
  4. scoring='roc_auc': Evaluation metric (ROC AUC score).
  5. random_state=42: Ensures reproducibility.
  6. disable_update_check=True: Disables update checks for TPOT.
  7. config_dict='TPOT light': Uses a predefined configuration for a  lighter set of models and preprocessing steps.
- #### Baseline Model with Logistic Regression
  1. LogisticRegression model is initialized.
  2. lr.fit(x_train, y_train) trains the Logistic Regression model on the training data.
  3. lr.predict(x_test) generates predictions for the test data.
  4. lr.score(x_test, y_test) calculates the accuracy of the   Logistic Regression model on the test data and prints it.
   5. The first few predictions are printed to provide a glimpse of the model's output.


## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- tpot

## Summary

The steps outlined in the code provide a comprehensive workflow for predicting blood donation behavior: 

- Load and prepare the data.
- Preprocess the data by scaling features.
- Separate features and the target variable.
- Split the data into training and testing sets.
- Train and evaluate models using TPOT for automated machine learning and Logistic Regression for a baseline comparison.

## Result

The TPOTClassifier will optimize machine learning pipelines and provide a model that maximizes the ROC AUC score on the test data. The LogisticRegression model serves as a baseline for comparison, with its accuracy printed alongside the first few predictions
