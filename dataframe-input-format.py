import joblib
import pandas as pd

# Load the pipeline object
pipeline = joblib.load('pipeline.joblib')

# Prepare new data as a pandas dataframe
new_data = pd.DataFrame({
    'Airline': ['Vistara'],
    'Classes': ['Business'],
    'Source': ['Ahmedabad'],
    'Departure': ['6 AM - 12 PM'],
    'Total_stops': ['1-stop'],
    'Arrival': ['After 6 PM'],
    'Destination': ['Chennai'],
    'Duration_in_hours': [13.0833],
    'Days_left': [50],
    'Year': [2023],
    'Month': [3],
    'Day': [6],
    'Day_of_week': [0]
})

# Predict using the pipeline
y_pred = pipeline.predict(new_data)
print(y_pred)
