import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# For reproducibility
np.random.seed(42)

# Locations ordered from highest to lowest price
locations = ['Lalitpur', 'Kathmandu', 'Bhaktapur', 'Pokhara', 'Biratnagar']
land_price_per_aana = [4000000, 3500000, 3000000, 2000000, 1500000]  # NPR per aana

# Create base dataset with realistic variations
data = pd.DataFrame({
    'Location': np.repeat(locations, 5),  # 5 samples per location
    'Rooms': np.random.randint(1, 6, size=25),  # 1-5 rooms
    'Area_sqft': np.random.randint(500, 3000, size=25),  # 500-3000 sqft
    'House_Type': ['House'] * 25
})

# Convert area to aana (1 aana = 342.25 sq.ft.)
data['Area_aana'] = data['Area_sqft'] / 342.25

# Map land price to location
price_map = dict(zip(locations, land_price_per_aana))
data['Land_Price_per_aana'] = data['Location'].map(price_map)

# Feature engineering - realistic pricing
construction_cost_per_sqft = 15000  # NPR
data['Construction_Cost'] = data['Area_sqft'] * construction_cost_per_sqft
data['Land_Cost'] = data['Area_aana'] * data['Land_Price_per_aana']
data['Price_NPR'] = data['Land_Cost'] + data['Construction_Cost']

# Add room premium (higher premium for more rooms)
data['Price_NPR'] += data['Rooms'] * 250000  # Adding NPR 250,000 per room

# Add location premium factor (extra premium for prime locations)
location_premium = {'Lalitpur': 1.2, 'Kathmandu': 1.15, 'Bhaktapur': 1.1, 'Pokhara': 1.05, 'Biratnagar': 1.0}
data['Price_NPR'] = data['Price_NPR'] * data['Location'].map(location_premium)

# Train the model
X = data[['Rooms', 'Area_aana', 'Land_Price_per_aana']]
y = data['Price_NPR']
model = LinearRegression()
model.fit(X, y)

# Prediction function
def predict_price(rooms, area_sqft, location):
    area_aana = area_sqft / 342.25
    land_price = price_map.get(location, 1500000)  # Default to Biratnagar if not found
    
    # Predict total price
    features = [[rooms, area_aana, land_price]]
    total_price = model.predict(features)[0]
    
    # Apply location premium
    total_price *= location_premium.get(location, 1.0)
    
    return total_price

# Streamlit app
st.title("Nepal House Price Predictor")
st.write("Enter house details to get estimated prices in NPR.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
    with col2:
        area_sqft = st.number_input("Area (sq. ft.)", min_value=100, value=1000)
    
    location = st.selectbox("Location", locations, index=1)  # Default to Kathmandu
    
    submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        total_price = predict_price(rooms, area_sqft, location)
        area_aana = area_sqft / 342.25
        
        st.subheader("Prediction Results")
        st.metric("Total Estimated Price", f"NPR {total_price:,.0f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Price per Room", f"NPR {total_price/rooms:,.0f}")
        with col2:
            st.metric("Price per Aana", f"NPR {total_price/area_aana:,.0f}")

# Add some explanation
st.markdown("""
**Note:** 
- 1 Aana = 342.25 sq.ft.
- Prices include both land and construction costs.
- Location pricing follows this order (highest to lowest): Lalitpur > Kathmandu > Bhaktapur > Pokhara > Biratnagar
- Model accounts for room count, area, and location premium.
- Prices are estimates based on current market trends.
""")

# Show sample data if checkbox is selected
if st.checkbox("Show sample data"):
    st.dataframe(data.sort_values('Price_NPR', ascending=False).head(10))
