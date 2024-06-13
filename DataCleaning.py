import pandas as pd
import numpy as np

# Load the combined data with low_memory=False to avoid DtypeWarning
combined_data_path = r"C:\Users\Ab Deshmukh\Desktop\Python\VSCode\RealEstate\Resale data.csv"
df = pd.read_csv(combined_data_path, low_memory=False)

# Step 1: Check for duplicates and remove them
print(f"Original dataset shape: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"Dataset shape after removing duplicates: {df.shape}")

# Step 2: Handle missing values
# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Fill or drop missing values as appropriate
# For this example, we will fill numerical columns with the median and categorical columns with the mode
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object]).columns

df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

print("Missing values after handling:")
print(df.isnull().sum())

# Step 4: Save the cleaned data
cleaned_data_path = "Ready.csv"
df.to_csv(cleaned_data_path, index=False)

print(f"Cleaned data saved to {cleaned_data_path}")
