import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
cleaned_data_path = r"C:\Users\Ab Deshmukh\Desktop\Python\VSCode\RealEstate\only.csv"
df = pd.read_csv(cleaned_data_path)

# Select columns of interest
columns_of_interest = ['transaction_year','town', 'flat_type', 'storey_level', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'resale_price']

# Calculate correlation matrix
correlation_matrix = df[columns_of_interest].corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.show()

# Print correlation values with respect to 'resale_price'
print("Correlation with respect to 'resale_price':")
print(correlation_matrix['resale_price'].sort_values(ascending=False))
