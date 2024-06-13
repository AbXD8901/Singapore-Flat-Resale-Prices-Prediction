import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load the cleaned data
cleaned_data_path = r"C:\Users\Ab Deshmukh\Desktop\Python\VSCode\RealEstate\only.csv"
df = pd.read_csv(cleaned_data_path)

# Define features and target variable
X = df[['transaction_year','town', 'flat_type', 'storey_level', 'floor_area_sqm', 'flat_model', 'lease_commence_date']]
y = df['resale_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models and their hyperparameters for tuning
models = {
    "Linear Regression": (LinearRegression(), {}),
    "Decision Tree": (DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    }),
    "Random Forest": (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    }),
    "XGBoost": (XGBRegressor(random_state=42), {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    })
}

# Train and evaluate models with hyperparameter tuning
results = {}
best_estimators = {}

for name, (model, params) in models.items():
    if params:
        grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }
    best_estimators[name] = best_model

# Find best performer
best_model_name = min(results, key=lambda k: results[k]['RMSE'])

# Print results
print("Performance Metrics:")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R2 Score: {metrics['R2 Score']:.2f}")
    print()

print(f"Best performer: {best_model_name}")

# Optionally, you can also print the best hyperparameters for the best model
print(f"Best hyperparameters for {best_model_name}:")
print(best_estimators[best_model_name])
