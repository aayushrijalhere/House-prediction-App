import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# For reproducibility
np.random.seed(42)

# Sample housing data for Kathmandu
locations = ['Tarkeshwor', 'Kageshwori Manohara', 'Kirtipur', 'Chandragiri']
land_price_per_aana = [1500000, 1800000, 2200000, 3000000]  # NPR per aana (constant for each location)

# Create more comprehensive dataset with multiple samples per location
num_samples_per_location = 20
data = pd.DataFrame()

for loc, price_per_aana in zip(locations, land_price_per_aana):
    # Generate random samples for each location
    loc_data = pd.DataFrame({
        'Location': [loc] * num_samples_per_location,
        'Rooms': np.random.randint(1, 6, num_samples_per_location),
        'Area_sqft': np.random.randint(800, 2500, num_samples_per_location),
        'House_Type': ['House'] * num_samples_per_location
    })
    
    # Convert area to aana (1 aana = 342.25 sq.ft.)
    loc_data['Area_aana'] = loc_data['Area_sqft'] / 342.25
    
    # Set land price per aana (constant for location)
    loc_data['Land_Price_per_aana'] = price_per_aana
    
    # Land cost calculation
    loc_data['Land_Cost'] = loc_data['Area_aana'] * loc_data['Land_Price_per_aana']
    
    # Price increases by (price_per_aana / 342.25) per 100 sq.ft
    loc_data['Area_Price_Increment'] = (loc_data['Area_sqft'] / 100) * (price_per_aana / 342.25)
    
    # Calculate electricity bill based on first 3 digits of Area_Price_Increment
    loc_data['Electricity_Bill'] = loc_data['Area_Price_Increment'].astype(str).str.replace('.', '').str[:3].astype(float)
    
    # Total price components (only land cost and electricity bill now)
    loc_data['Price_NPR'] = loc_data['Land_Cost'] + loc_data['Electricity_Bill']

# Create separate models for each location
location_models = {}
for loc in locations:
    loc_data = data[data['Location'] == loc]
    X = loc_data[['Rooms', 'Area_sqft']]
    y = loc_data['Electricity_Bill']  # Now only modeling electricity bill
    model = LinearRegression()
    model.fit(X, y)
    location_models[loc] = model

# Prediction function
def predict_prices(rooms, area_sqft, location):
    # Get land price per aana for this location
    price_per_aana = dict(zip(locations, land_price_per_aana))[location]
    
    # Calculate land cost
    land_cost = (area_sqft / 342.25) * price_per_aana
    
    # Calculate area price increment
    area_increment = (area_sqft / 100) * (price_per_aana / 342.25)
    
    # Calculate electricity bill from first 3 digits of area increment (without decimal)
    electricity_bill = float(f"{area_increment:.2f}".replace('.', '')[:3])
    
    # Total price
    total_price = land_cost + electricity_bill
    
    # Price per room (only electricity bill divided by rooms, land cost excluded)
    price_per_room = electricity_bill / rooms if rooms > 0 else 0
    
    return price_per_room, price_per_aana, area_increment, electricity_bill, total_price

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
        try:
            price_per_room, price_per_aana, area_increment, electricity_bill, total_price = predict_prices(rooms, area_sqft, location)
            
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price per Room (electricity only)", f"NPR {price_per_room:,.0f}")
            with col2:
                st.metric("Price per Aana", f"NPR {price_per_aana:,.0f}")
            
            st.subheader("Detailed Breakdown")
            st.write(f"- Total property price: NPR {total_price:,.0f}")
            st.write(f"  - Land cost: NPR {(area_sqft / 342.25) * price_per_aana:,.0f}")
            st.write(f"  - Electricity bill: NPR {electricity_bill:,.0f}")
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Add some explanation
st.markdown("""
**Note:** 
- 1 Aana = 342.25 sq.ft.
- Price per aana is constant for each location.
- Price per room includes only the electricity bill component divided by number of rooms
- Electricity bill is calculated using first 3 digits of area price increment
- Total price includes:
  - Land cost (based on area in aana)
  - Electricity bill (from area increment)
""")
