import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# For reproducibility
np.random.seed(42)

# Sample housing data for Kathmandu
locations = ['Tarkeshwor', 'Kageshwori Manohara', 'Kirtipur', 'Chandragiri']
land_price_per_aana = [1500000, 1800000, 2200000, 3000000]  # NPR per aana (constant for each location)

# Create base dataset
data = pd.DataFrame({
    'Location': locations,
    'Rooms': [3, 4, 5, 6],
    'Area_sqft': [1000, 1200, 1600, 2000],
    'House_Type': ['House'] * 4
})

# Convert area to aana (1 aana = 342.25 sq.ft.)
data['Area_aana'] = data['Area_sqft'] / 342.25

# Map land price to location (constant per aana for each location)
price_map = dict(zip(locations, land_price_per_aana))
data['Land_Price_per_aana'] = data['Location'].map(price_map)

# Feature engineering - realistic pricing
construction_cost_per_sqft = 15000  # NPR
data['Construction_Cost'] = data['Area_sqft'] * construction_cost_per_sqft
data['Land_Cost'] = data['Area_aana'] * data['Land_Price_per_aana']
data['Price_NPR'] = data['Land_Cost'] + data['Construction_Cost']

# Room pricing - price per room decreases as number of rooms increases
# Using a logarithmic function to model diminishing returns
data['Room_Premium'] = np.log(data['Rooms'] + 1) * 500000  # Logarithmic scaling
data['Price_NPR'] += data['Room_Premium']

# Train the model
X = data[['Rooms', 'Area_aana', 'Land_Price_per_aana']]
y = data['Price_NPR']
model = LinearRegression()
model.fit(X, y)

# Prediction function
def predict_prices(rooms, area_sqft, location):
    # Convert area to aana
    area_aana = area_sqft / 342.25
    land_price = price_map.get(location, 1800000)  # Default value
    
    # Predict total price
    features = [[rooms, area_aana, land_price]]
    total_price = model.predict(features)[0]
    
    # Calculate price per room and per aana
    price_per_room = total_price / rooms
    price_per_aana = land_price  # Constant per aana for location
    
    return price_per_room, price_per_aana

# Streamlit app
st.title("Kathmandu House Price Predictor")
st.write("Enter house details to get estimated prices in NPR.")

# Input form
with st.form("prediction_form"):
    rooms = st.number_input("Number of Rooms", min_value=1, value=3)
    area_sqft = st.number_input("Area (sq. ft.)", min_value=100, value=1000)
    location = st.selectbox("Location", locations)
    
    submitted = st.form_submit_button("Predict Prices")
    
    if submitted:
        price_per_room, price_per_aana = predict_prices(rooms, area_sqft, location)
        
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Price per Room", f"NPR {price_per_room:,.0f}")
        with col2:
            st.metric("Price per Aana", f"NPR {price_per_aana:,.0f}")

# Add some explanation
st.markdown("""
**Note:** 
- 1 Aana = 342.25 sq.ft.
- Price per aana is constant for each location.
- Price per room decreases as number of rooms increases.
- Price per room increases as area increases.
- Prices include both land and construction costs.
- Model is trained on sample data and provides estimates only.
""")
