import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scalers
with open('finalmodel.pkl', 'rb') as f:
    finalmodel = pickle.load(f)
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Streamlit app title
st.title("Doctor Consultation Fee Prediction")

# Input fields for the user to provide data
experience = st.number_input("Years of Experience", min_value=0, max_value=66, step=1)
profile = st.selectbox("Doctor Specialization", ["Ayurveda", "Dentist", "Dermatologists", "ENT Specialist", "General Medicine", "Homeopath"])
place = st.selectbox("Place", ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Coimbatore", "Ernakulam", "Thiruvananthapuram"])

# Mapping Profile and Place to match the encoding in the training phase
profile_mapping = {"Ayurveda": "profile_Ayurveda", "Dentist": "profile_Dentist", "Dermatologists": "profile_Dermatologists", 
                   "ENT Specialist": "profile_ENT Specialist", "General Medicine": "profile_General Medicine", "Homeopath": "profile_Homeopath"}

place_mapping = {"Bangalore": "place_Bangalore", "Mumbai": "place_Mumbai", "Delhi": "place_Delhi", "Hyderabad": "place_Hyderabad", 
                 "Chennai": "place_Chennai", "Coimbatore": "place_Coimbatore", "Ernakulam": "place_Ernakulam", 
                 "Thiruvananthapuram": "place_Thiruvananthapuram"}

# Prepare the input DataFrame for prediction
input_data = pd.DataFrame(columns=scaler_X.feature_names_in_)
input_data.loc[0] = 0  # Initialize with zeros

# Set the input features based on user input
input_data["Experience"] = np.log(experience) if experience > 0 else 0
input_data[profile_mapping[profile]] = 1
input_data[place_mapping[place]] = 1

# Button to trigger prediction
if st.button("Predict"):
    # Scale the input features
    input_data_scaled = scaler_X.transform(input_data)

    # Make the prediction
    prediction = finalmodel.predict(input_data_scaled)

    # Reverse the scaling for the prediction
    prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1))

    # Display the prediction
    st.write(f"The predicted consultation fee is: {prediction[0][0]:.2f}")
