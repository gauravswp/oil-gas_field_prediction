import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load data for dropdown options and to fit LabelEncoders
df2 = pd.read_csv("oil_test.csv")  # Replace with the actual path to your training dataset

# Initialize LabelEncoders for each categorical column
field_name_encoder = LabelEncoder()
reservoir_unit_encoder = LabelEncoder()
country_encoder = LabelEncoder()
basin_name_encoder = LabelEncoder()
operator_company_encoder = LabelEncoder()

# Fit encoders on the training data's unique values for consistency
field_name_encoder.fit(df2['Field name'])
reservoir_unit_encoder.fit(df2['Reservoir unit'])
country_encoder.fit(df2['Country'])
basin_name_encoder.fit(df2['Basin name'])
operator_company_encoder.fit(df2['Operator company'])

# Set up dropdown options using unique values
field_names = df2['Field name'].unique()
reservoir_units = df2['Reservoir unit'].unique()
countries = df2['Country'].unique()
basin_names = df2['Basin name'].unique()
operator_companies = df2['Operator company'].unique()

# Streamlit app layout
st.title("Oil and Gas Field Prediction App")

# Dropdowns for user input
field_name = st.selectbox("Field Name", field_names)
reservoir_unit = st.selectbox("Reservoir Unit", reservoir_units)
country = st.selectbox("Country", countries)
basin_name = st.selectbox("Basin Name", basin_names)
operator_company = st.selectbox("Operator Company", operator_companies)

# Check if selected fields match a valid row in the training data
valid_input = df2[
    (df2['Field name'] == field_name) &
    (df2['Reservoir unit'] == reservoir_unit) &
    (df2['Country'] == country) &
    (df2['Basin name'] == basin_name) &
    (df2['Operator company'] == operator_company)
]

if st.button("Predict"):
    if valid_input.empty:
        st.write("Incorrect data: Selected fields do not match any valid record.")
    else:
        # Encode user inputs using the trained LabelEncoders
        encoded_field_name = field_name_encoder.transform([field_name])[0]
        encoded_reservoir_unit = reservoir_unit_encoder.transform([reservoir_unit])[0]
        encoded_country = country_encoder.transform([country])[0]
        encoded_basin_name = basin_name_encoder.transform([basin_name])[0]
        encoded_operator_company = operator_company_encoder.transform([operator_company])[0]

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            "Field name": [encoded_field_name],
            "Reservoir unit": [encoded_reservoir_unit],
            "Country": [encoded_country],
            "Basin name": [encoded_basin_name],
            "Operator company": [encoded_operator_company],
            "Latitude": [0],  # Placeholder or actual value if available
            "Longitude": [0]  # Placeholder or actual value if available
        })

        # Make prediction
        prediction = model.predict(input_data)
        
        # Interpret the prediction output
        result = "ONSHORE" if prediction[0] == 1 else "OFFSHORE"
        st.write(f"The predicted location type is: **{result}**")
