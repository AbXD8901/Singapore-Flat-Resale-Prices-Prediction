import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import base64

# Function to encode image to base64
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your local image
image_path = r'C:\Userse\image.jpg'
base64_image = get_base64_encoded_image(image_path)

# Set background image using base64 encoded string
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the cleaned data
cleaned_data_path = r"C:\Users\one.csv"
df = pd.read_csv(cleaned_data_path)

# Define features and target variable
X = df[['transaction_year', 'town', 'flat_type', 'storey_level', 'floor_area_sqm', 'flat_model', 'lease_commence_date']]
y = df['resale_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBRegressor model
model = XGBRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Streamlit Application
st.title("Flat Resale Price Predictor")

# User inputs
town_options = sorted(['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
flat_type_options = sorted(['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION'])
flat_model_options = sorted(['2-room', '3gen', 'adjoined flat', 'apartment', 'dbss', 'improved','improved-maisonette', 'maisonette', 'model a', 'model a-maisonette', 'model a2', 'multi generation', 'new generation', 'premium apartment','premium apartment loft', 'premium maisonette', 'simplified', 'standard','terrace', 'type s1', 'type s2'])

transaction_year = st.number_input("Transaction Year", min_value=1990, max_value=datetime.now().year, value=datetime.now().year)
town = st.selectbox("Select Town", town_options)
flat_type = st.selectbox("Select Flat Type", flat_type_options)
flat_model = st.selectbox("Select Flat Model", flat_model_options)

storey_level = st.number_input("Storey Level", min_value=1, max_value=40, value=20)
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=200, value=100)
lease_commence_date = st.number_input("Lease Commence Date", min_value=1960, max_value=datetime.now().year, value=2000)

# Button to trigger prediction
if st.button("Predict Resale Price"):
    # Map input values to encoded values
    town_encoded = town_options.index(town)
    flat_type_encoded = flat_type_options.index(flat_type)
    flat_model_encoded = flat_model_options.index(flat_model)

    # Prepare the user input for prediction
    user_input = np.array([transaction_year, town_encoded, flat_type_encoded, storey_level, floor_area_sqm, flat_model_encoded, lease_commence_date]).reshape(1, -1)

    # Make prediction using the XGBRegressor model
    prediction = model.predict(user_input)[0]

    # Display prediction to the user with increased size
    st.success(f"Predicted Resale Price: **${prediction:.2f}**")

    # Calculate age of property and remaining lease
    current_year = datetime.now().year
    age_of_property = current_year - lease_commence_date
    remaining_lease = 99 - age_of_property

    st.info(f"Age of Property: {age_of_property} years")
    st.info(f"Remaining Lease: {remaining_lease} years")
