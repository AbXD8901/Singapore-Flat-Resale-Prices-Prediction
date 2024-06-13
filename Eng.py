import pandas as pd
import numpy as np

# Load the cleaned data
cleaned_data_path = r"C:\Users\Ab Deshmukh\Desktop\Python\VSCode\RealEstate\Ready.csv"
df = pd.read_csv(cleaned_data_path)

# Step 1: Extract Year and Month from the 'month' column
df['transaction_year'] = pd.to_datetime(df['month']).dt.year
df['transaction_month'] = pd.to_datetime(df['month']).dt.month

# Step 2: Convert 'storey_range' to numerical values (average of the range)
def storey_to_numeric(storey_range):
    if '-' in storey_range:
        start, end = map(int, storey_range.split(' TO '))
        return int((start + end) / 2)
    elif 'TO' in storey_range:
        start, end = map(int, storey_range.split(' TO '))
        return int((start + end) / 2)
    return int(storey_range.split(' ')[0])

df['storey_level'] = df['storey_range'].apply(storey_to_numeric)

# Ensure relevant columns and created features are included
relevant_columns = ['transaction_year', 'transaction_month', 'town', 'flat_type', 'storey_level', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'resale_price']

# Check for missing values and print the result
print(df[relevant_columns].isnull().sum())

# Display the first few rows of the updated DataFrame
print(df[relevant_columns].head())

# Save the DataFrame with the new features
enhanced_data_path = "FINAL.csv"
df[relevant_columns].to_csv(enhanced_data_path, index=False)

print(f"Enhanced data saved to {enhanced_data_path}")
