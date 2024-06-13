import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the cleaned data
cleaned_data_path = r'C:\Users\Ab Deshmukh\Desktop\Python\VSCode\RealEstate\MAIN.csv'
df = pd.read_csv(cleaned_data_path)

# Select categorical columns for label encoding
categorical_cols = ['town', 'flat_type', 'flat_model']

# Make a copy of the original dataframe
df_encoded = df.copy()

# Initialize label encoder
label_encoders = {}

# Label encode categorical columns and save label mappings
label_mappings = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df_encoded[col] = label_encoders[col].fit_transform(df[col])
    label_mappings[col] = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))

# Save label mappings to a text file
with open('label_mappings.txt', 'w') as file:
    for col, mapping in label_mappings.items():
        file.write(f"{col}:\n")
        for label, value in mapping.items():
            file.write(f"{label}: {value}\n")
        file.write("\n")

# Save the encoded dataset
encoded_data_path = "only.csv"
df_encoded.to_csv(encoded_data_path, index=False)

print(f"Encoded data saved to {encoded_data_path}")
