import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the model
with open("C:\\Users\\roopt\\Car_Dekho\\tuned_random_forest_model_copy.pkl", 'rb') as f:
    model = pickle.load(f)

# Feature names
features = ['mileage_km', 'number_owner', 'mileage', 'engine_power', 'torque_car', 'Max Power',
            'seats_car', 'age_of_car', 'fuel_type', 'body_type', 'transmission', 'oem']

# Categorical variable mappings
categorical_mappings = {
    'fuel_type': {'Petrol': 0, 'Diesel': 1, 'LPG': 4, 'CNG': 2, 'Electric': 3},
    'body_type': {'Hatchback': 0, 'SUV': 1, 'Sedan': 2, 'MUV': 3, 'Coupe': 5,
                  'Minivans': 4, 'Pickup Trucks': 6, 'Convertibles': 7, 'Hybrids': 9, 'Wagon': 10},
    'transmission': {'Automatic': 1, 'Manual': 0},
    'oem': {'Maruti': 0, 'Ford': 1, 'Tata': 2, 'Hyundai': 3, 'Jeep': 4, 'Datsun': 5, 'Honda': 6, 'Mahindra': 7,
    'Mercedes-Benz': 8, 'BMW': 9, 'Renault': 10, 'Audi' :11, 'Toyota': 12, 'Mini': 13, 'Kia': 14, 'Skoda': 15, 'Volkswagen': 16,
    'Volvo': 17, 'MG': 18, 'Nissan': 19, 'Fiat': 20, 'Mahindra Ssangyong': 21, 'Mitsubishi': 22, 'Jaguar': 23,
    'Land Rover': 24, 'Chevrolet': 25, 'Citroen': 26, 'Opel': 27, 'Mahindra Renault': 28, 'Hindustan Motors': 29, 'Porsche': 30,
    'Isuzu': 31, 'Lexus': 32}
}

# Input widgets for user interaction
st.title("Car Price Prediction App with OEM Comparison")

input_data = {}
for feature in features:
    if feature in categorical_mappings:
        selected_option = st.sidebar.selectbox(f"Select {feature.capitalize()}:", options=list(categorical_mappings[feature].keys()))
        input_data[feature] = categorical_mappings[feature][selected_option]
    else:
        input_data[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').capitalize()}:")

# Predict price and generate comparison chart
if st.sidebar.button("Predict"):
    # Prepare input array
    input_array = np.array([input_data[feature] for feature in features]).reshape(1, -1)
    prediction = model.predict(input_array)

    # Display the selected feature values
    st.subheader("Selected Feature Values:")
    for feature, value in input_data.items():
        st.write(f"{feature.replace('_', ' ').capitalize()}: {value}")

    # Display the prediction result
    st.subheader("Prediction Result:")
    st.write(f"The estimated car price is: INR {prediction[0]:,.2f}")

    # Comparison chart for all OEMs
    st.subheader("Comparison of Price Predictions Across OEMs:")
    oem_predictions = {}
    for oem_name, oem_code in categorical_mappings['oem'].items():
        input_array[0, -1] = oem_code  # Change only the 'oem' value
        oem_predictions[oem_name] = model.predict(input_array)[0]

    # Plot the comparison as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(oem_predictions.keys(), oem_predictions.values(), color='skyblue')
    plt.xlabel("OEM")
    plt.ylabel("Predicted Price (INR)")
    plt.title("Price Prediction Comparison Across OEMs")
    st.pyplot(plt)
