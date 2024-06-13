# Singapore-Flat-Resale-Prices-Prediction
The objective of this project is to develop a machine learning model (here used XGBRegressor) and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. 
Real Estate Resale Price Prediction
This project aims to predict the resale prices of flats using historical data and machine learning models. The model uses features such as transaction year, town, flat type, storey level, floor area, flat model, and lease commence date to predict the resale price. The primary model used in this project is the XGBoost Regressor.

## Table of Contents
Introduction
Dataset
Data Preprocessing
Model Training
Evaluation
Feature Importance
Usage
Results
Care and Considerations

## Dataset
The dataset used for this project contains information on flat resale transactions from 1990 to 2024. The dataset includes the following columns:
transaction_year: The year of the transaction.
town: The town where the flat is located.
flat_type: The type of flat.
storey_level: The storey level of the flat.
floor_area_sqm: The floor area of the flat in square meters.
flat_model: The model of the flat.
lease_commence_date: The year the lease commenced.
resale_price: The resale price of the flat.

## Data Preprocessing
Data preprocessing involves cleaning and encoding the data to make it suitable for model training. Key steps include:
### Standardizing Flat Model Labels: Ensure consistency by converting all flat model labels to lowercase.
Label Encoding: Encode categorical variables into numerical values for model training. In this project, label encoding was already performed in the dataset.
Special care was taken to handle inconsistencies in the dataset. For example, instances of 'MULTI-GENERATION' and 'MULTI GENERATION' in the flat_type column were standardized.
i used is the XGBoost Regressor. The dataset is split into training and testing sets to evaluate model performance.

## Initialize and train the model
from xgboost import XGBRegressor
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
Evaluation
The model's performance is evaluated using Mean Squared Error (MSE) and R2 Score. The MSE gives an idea of how much the predictions deviate from the actual values, while the R2 Score indicates how well the model explains the variance in the data.

## Usage
To use the model for predicting resale prices:

## Load the preprocessed dataset.
Split the data into training and testing sets.
Train the model using the training data.
Use the model to make predictions on new data.
Results
### The model achieved the following performance metrics:
Mean Squared Error (MSE): 847,560,309.02
R2 Score: 0.97
These metrics indicate that the model has high accuracy in predicting flat resale prices.

## Care and Considerations
Data Quality: Ensure that the dataset is clean and free from inconsistencies. Handle missing values and outliers appropriately.
Feature Engineering: Consider additional features that might impact resale prices, such as proximity to amenities, age of the property, and remaining lease period.
Model Evaluation: Continuously evaluate the model's performance on new data to ensure its accuracy.
Hyperparameter Tuning: Experiment with different hyperparameters to find the optimal configuration for the model.
Regular Updates: Update the model regularly with new data to maintain its accuracy over time.
