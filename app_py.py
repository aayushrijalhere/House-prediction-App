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
    
    # Room pricing - price per room decreases as number of rooms increases
    loc_data['Room_Premium'] = np.log(loc_data['Rooms'] + 1) * 500000  # Logarithmic scaling
    
    # Total price 
    loc_data['Price_NPR'] = (loc_data['Land_Cost'] + 
    loc_data['Area_Price_Increment'] + 
                            loc_data['Electricity_Bill'] +
    loc_data['Room_Premium'])
    
    data = pd.concat([data, loc_data])

# Create separate models for each location
location_models = {}
for loc in locations:
    loc_data = data[data['Location'] == loc]
    X = loc_data[['Rooms', 'Area_sqft']]
    y = loc_data['Price_NPR']
    model = LinearRegression()
    model.fit(X, y)
    location_models[loc] = model

# Prediction function
def predict_prices(rooms, area_sqft, location):
    # Get the model for this location
    model = location_models.get(location)
    if model is None:
        raise ValueError(f"No model found for location: {location}")
    
    # Predict total price
    features = [[rooms, area_sqft]]
    total_price = model.predict(features)[0]
    
    # Get land price per aana for this location
    price_per_aana = dict(zip(locations, land_price_per_aana))[location]
    
    # Calculate area price increment
    area_increment = (area_sqft / 100) * (price_per_aana / 342.25)
    
    # Calculate electricity bill from first 3 digits of area increment (without decimal)
    electricity_bill = float(f"{area_increment:.2f}".replace('.', '')[:3])
    
    # Calculate price per room including only area increment and electricity bill
    total_increment_and_bill = area_increment + electricity_bill
    price_per_room = total_increment_and_bill / rooms if rooms > 0 else 0
    price_per_aana = price_per_aana  # Constant per aana for location
    
    return price_per_room, price_per_aana, area_increment, electricity_bill

# Streamlit app
st.title("Kathmandu Rent Price Predictor")
st.write("Enter house details to get estimated prices in NPR.")

# Input form
with st.form("prediction_form"):
    rooms = st.number_input("Number of Rooms", min_value=1, value=3)
    area_sqft = st.number_input("Area (sq. ft.)", min_value=100, value=1000)
    location = st.selectbox("Location", locations)
    
    submitted = st.form_submit_button("Predict Prices")
    
    if submitted:
        try:
            price_per_room, price_per_aana, area_increment, electricity_bill = predict_prices(rooms, area_sqft, location)
            
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price per Room (Monthly)", f"NPR {price_per_room:,.0f}")
            with col2:
                st.metric("Price per Aana", f"NPR {price_per_aana:,.0f}")
            
            st.subheader("Price Components")
            st.write(f"- Area price increment (for {area_sqft} sq.ft.): NPR {area_increment:,.0f}")
            st.write(f"- Electricity bill (first 3 digits of area increment): NPR {electricity_bill:,.0f}")
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Add some explanation
st.markdown("""
**Note:** 
- 1 Aana = 342.25 sq.ft.
- Prices include:
  - Area price increment
  - Electricity bill (from area increment)
""")
