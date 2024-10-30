import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import pickle

# Load data
df1 = pd.read_csv("oil_test.csv")  # Test data
df2 = pd.read_csv("train_oil.csv")  # Train data

# Initialize Nominatim API for geolocation
geolocator = Nominatim(user_agent="MyApp")

import time
from geopy.exc import GeocoderTimedOut

# Retry function for geolocation with 3 retries and a 5-second pause
def geocode_with_retry(location, geolocator, retries=3):
    for attempt in range(retries):
        try:
            return geolocator.geocode(location)
        except GeocoderTimedOut:
            if attempt < retries - 1:
                time.sleep(5)  # wait before retrying
            else:
                print(f"Geocoding for '{location}' failed after {retries} retries.")
                return None
            
# Location mappings for df1 and df2
locations_df1 = {
    "JULY": "GULF OF SUEZ",
    "DJEITUN": "RED SERIES",
    "ARUN": "ARUN",
    "BALOL": "KALOL",
    "CHALYBEAT SPRINGS": "SMACKOVER",
    "GORGON": "MUNGAROO",
    "KG": "KG",
    "KHALDA": "KHALDA",
    "LIUHUA 11-1": "ZHUJIANG",
    "MAYDAN MAHZAM": "Qatar",
    "WEIYUAN": "WEIYUAN",
    "VERMEJO-MOORE HOOPER": "FUSSELMAN",
    "RAMA": "BATURAJA",
    "PRIRAZLOM": "TIMAN-PECHORA",
    "OCTOBER": "GULF OF SUEZ",
    "BELAYIM MARINE": "GULF OF SUEZ"
}

locations_df2 = {
    "BADR EL DIN-2": "BAHARIYA",
    "ZAKUM": "ZAKUM",
    "UZEN": "UZEN",
    "SCOTT": "SCOTT",
    "BRIDGER LAKE": "BRIDGER LAKE",
    "YAKIN": "YAKIN",
    "BARQUE": "BARQUE",
    "ORENBURG": "ORENBURG",
    "CASHIRIARI": "CASHIRIARI",
    "ROURKE GAP": "MINNELUSA",
    "ANDREW": "ANDREW SANDSTONE",
    "QARUN": "BAHARIYA",
    "TALCO": "PALUXY",
    "BEAVER LODGE": "BEAVER LODGE",
    "ALPINE": "ALPINE",
    "RHOURDE EL BAGUEL": "GHADAMES",
    "NORTH ROBERTSON": "NORTH ROBERTSON",
    "CAROLINE": "SWAN HILLS",
    "TABER NORTH": "TABER NORTH",
    "GASIKULE": "GASIKULE",
    "EMPIRE ABO": "ABO",
    "TIA JUANA": "TIA JUANA",
    "HARMATTAN-ELKTON": "TURNER VALLEY",
    "GLENBURN": "GLENBURN",
    "OROCUAL": "OROCUAL",
    "WUBAITI": "HUANGLONG",
    "PALM VALLEY": "PALM VALLEY",
    "YIBAL": "YIBAL",
    "ULA": "ULA",
    "WEST SEMINOLE": "WEST SEMINOLE"
}

# Apply geocoding to df1
for field, location in locations_df1.items():
    geo_data = geocode_with_retry(location, geolocator)
    if geo_data:
        df1.loc[df1['Field name'] == field, 'Longitude'] = geo_data.longitude
        df1.loc[df1['Field name'] == field, 'Latitude'] = geo_data.latitude
    else:
        print(f"Could not retrieve geolocation for '{field}' in df1.")

# Apply geocoding to df2
for field, location in locations_df2.items():
    geo_data = geocode_with_retry(location, geolocator)
    if geo_data:
        df2.loc[df2['Field name'] == field, 'Longitude'] = geo_data.longitude
        df2.loc[df2['Field name'] == field, 'Latitude'] = geo_data.latitude
    else:
        print(f"Could not retrieve geolocation for '{field}' in df2.")


# Separate numerical and object columns in training data
float_columns = df2.select_dtypes(include=['float'])
obj_columns = df2.select_dtypes(include=['object'])

# Drop unnecessary columns in training and test data
def drop_col(df):
    return df.drop(["Field name", "Hydrocarbon type", "Reservoir unit", "Country", "Region", "Reservoir period",
                    "Basin name", "Operator company", "Tectonic regime", "Lithology", "Structural setting",
                    "Reservoir status"], axis=1)

df2 = drop_col(df2)
df1 = drop_col(df1)

# Encode 'Onshore/Offshore' column in training data
le = LabelEncoder()
df2["Onshore/Offshore"] = le.fit_transform(df2["Onshore/Offshore"])

# Fill missing Latitude and Longitude values
df2["Latitude"] = df2["Latitude"].fillna(0)
df2["Longitude"] = df2["Longitude"].fillna(0)

# Standardize the data
scaler = StandardScaler()
df1 = scaler.fit_transform(df1)
X = scaler.fit_transform(df2.drop(columns="Onshore/Offshore"))

# Train-test split
y = df2["Onshore/Offshore"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize and train XGBoost classifier
xgb_clf = XGBClassifier(random_state=200, seed=100)
xgb_clf.fit(X_train, y_train)

# Evaluate and predict with the XGBoost classifier
y_pred = xgb_clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Prediction on test data
y_pred_test = xgb_clf.predict(df1)
df_result = pd.DataFrame({"Onshore/Offshore": y_pred_test})
df_result.index.name = "index"
df_result["Onshore/Offshore"] = df_result["Onshore/Offshore"].replace({0: "OFFSHORE", 1: "ONSHORE", 2: "ONSHORE/OFFSHORE"})

# y_pred=xgb_clf.predict(df1)
# print(y_pred)

# Save results to CSV
# df_result.to_csv("subm.csv")

# Save the trained XGBoost model to a pickle file
with open("model.pkl", "wb") as file:
    pickle.dump(xgb_clf, file)

# # Save the StandardScaler as well
# with open("scaler.pkl", "wb") as scaler_file:
#     pickle.dump(scaler, scaler_file)

# print("Model and scaler saved.")

# Load the saved XGBoost model from the pickle file
# with open(" model.pkl", "rb") as file:
#     loaded_model = pickle.load(file)

# # Load the saved StandardScaler
# with open("scaler.pkl", "rb") as scaler_file:
#     loaded_scaler = pickle.load(scaler_file)

# # Standardize the test data using the loaded scaler
# df1_scaled = loaded_scaler.transform(df1)

# # Make predictions using the loaded model
# y_pred_loaded = loaded_model.predict(df1_scaled)

# # Prepare the results
# df_result_loaded = pd.DataFrame({"Onshore/Offshore": y_pred_loaded})
# df_result_loaded.index.name = "index"
# df_result_loaded["Onshore/Offshore"] = df_result_loaded["Onshore/Offshore"].replace({0: "OFFSHORE", 1: "ONSHORE", 2: "ONSHORE/OFFSHORE"})

# # Display the predictions
# print(df_result_loaded)


# with open('oil_gas.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)


# try:
#     with open('oil_gas.pkl', "rb") as f:
#         geocoded_data = pickle.load(f)
#     print("Geocoded data loaded successfully.")
# except FileNotFoundError:
#     print("Pickle file not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")

# # Display the loaded geocoded data
# print(geocoded_data)
# Load the saved XGBoost model from the pickle file
# with open("oil_gas.pkl", "rb") as file:
#     loaded_model = pickle.load(file)


# df1_scaled = scaler.transform(df1)

# # Make predictions using the loaded model
# y_pred_loaded = loaded_model.predict(df1_scaled)

# # Prepare the results
# df_result_loaded = pd.DataFrame({"Onshore/Offshore": y_pred_loaded})
# df_result_loaded.index.name = "index"
# df_result_loaded["Onshore/Offshore"] = df_result_loaded["Onshore/Offshore"].replace({0: "OFFSHORE", 1: "ONSHORE", 2: "ONSHORE/OFFSHORE"})

# # Display the predictions
# print(df_result_loaded)