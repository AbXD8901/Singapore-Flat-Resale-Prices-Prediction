import pandas as pd

# Load your dataset
cleaned_data_path = r"C:\Users\Ab Deshmukh\FINAL.csv"
df = pd.read_csv(cleaned_data_path)

# Display unique values in the 'flat_model' column to identify variations
unique_flat_models = df['flat_model'].unique()
print("Unique flat_model values before standardization:")
print(unique_flat_models)

# Define a mapping to standardize categories
standardize_mapping = {
    '2-ROOM': '2-room',
    '3Gen': '3gen',
    'APARTMENT': 'apartment',
    'Adjoined flat': 'adjoined flat',
    'DBSS': 'dbss',
    'IMPROVED': 'improved',
    'IMPROVED-MAISONETTE': 'improved-maisonette',
    'Improved': 'improved',
    'Improved-Maisonette': 'improved-maisonette',
    'MAISONETTE': 'maisonette',
    'MODEL A': 'model a',
    'MODEL A-MAISONETTE': 'model a-maisonette',
    'MULTI GENERATION': 'multi generation',
    'Maisonette': 'maisonette',
    'Model A': 'model a',
    'Model A-Maisonette': 'model a-maisonette',
    'Model A2': 'model a2',
    'Multi Generation': 'multi generation',
    'NEW GENERATION': 'new generation',
    'New Generation': 'new generation',
    'PREMIUM APARTMENT': 'premium apartment',
    'Premium Apartment': 'premium apartment',
    'Premium Apartment Loft': 'premium apartment loft',
    'Premium Maisonette': 'premium maisonette',
    'SIMPLIFIED': 'simplified',
    'STANDARD': 'standard',
    'Simplified': 'simplified',
    'Standard': 'standard',
    'TERRACE': 'terrace',
    'Terrace': 'terrace',
    'Type S1': 'type s1',
    'Type S2': 'type s2'
}

# Apply the standardization mapping to 'flat_model' column
df['flat_model'] = df['flat_model'].map(standardize_mapping)

# Display unique values again to verify standardization
unique_flat_models_after = df['flat_model'].unique()
print("\nUnique flat_model values after standardization:")
print(unique_flat_models_after)

# Save the updated DataFrame back to CSV if needed
updated_data_path = r"C:\Users\Ab Deshmukh\Desktop\Python\VSCode\RealEstate\MAIN.csv"
df.to_csv(updated_data_path, index=False)
print(f"\nData saved to {updated_data_path}")
