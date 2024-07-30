# Save this as app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('number_of_casualties_model.pkl')

# Define the UI
st.title('Number of Casualties Prediction')
st.write("Please input the details of the accident:")

# Define input fields for all required features
time = st.text_input('Time', '12:00')
day_of_week = st.selectbox('Day of Week', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
age_band_of_driver = st.selectbox('Age Band of Driver', options=['18-30', '31-50', 'Over 51', 'Unknown'])
sex_of_driver = st.selectbox('Sex of Driver', options=['Male', 'Female', 'Unknown'])
educational_level = st.selectbox('Educational Level', options=['Above high school', 'Elementary school', 'Junior high school', 'High school', 'Unknown'])
vehicle_driver_relation = st.selectbox('Vehicle Driver Relation', options=['Owner', 'Employee', 'Unknown'])
driving_experience = st.selectbox('Driving Experience', options=['1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Unknown'])
type_of_vehicle = st.selectbox('Type of Vehicle', options=['Automobile', 'Lorry', 'Public (> 45 seats)', 'Others', 'Unknown'])
owner_of_vehicle = st.selectbox('Owner of Vehicle', options=['Owner', 'Governmental', 'Others', 'Unknown'])
service_year_of_vehicle = st.selectbox('Service Year of Vehicle', options=['< 1yr', '1-2yrs', '2-5yrs', '5-10yrs', 'Above 10yr', 'Unknown'])
defect_of_vehicle = st.selectbox('Defect of Vehicle', options=['Brakes', 'Lighting system', 'Steering', 'Other', 'Unknown'])
area_accident_occured = st.selectbox('Area Accident Occured', options=['Residential areas', 'Office areas', 'Commercial areas', 'Industrial areas', 'Others'])
lanes_or_medians = st.selectbox('Lanes or Medians', options=['One way', 'Two way', 'Others'])
road_allignment = st.selectbox('Road Alignment', options=['Straight road', 'T Junction', 'Crossing', 'Others'])
types_of_junction = st.selectbox('Types of Junction', options=['No junction', 'Y Junction', 'T Junction', 'Crossing', 'Others'])
road_surface_type = st.selectbox('Road Surface Type', options=['Asphalt roads', 'Earth roads', 'Gravel roads', 'Others'])
road_surface_conditions = st.selectbox('Road Surface Conditions', options=['Dry', 'Wet', 'Snowy', 'Others'])
light_conditions = st.selectbox('Light Conditions', options=['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit', 'Others'])
weather_conditions = st.selectbox('Weather Conditions', options=['Normal', 'Raining', 'Foggy', 'Other'])
type_of_collision = st.selectbox('Type of Collision', options=['Vehicle with vehicle collision', 'Vehicle with pedestrian', 'Other'])
number_of_vehicles_involved = st.number_input('Number of Vehicles Involved', min_value=1, max_value=5, value=1)
vehicle_movement = st.selectbox('Vehicle Movement', options=['Going straight', 'U-turn', 'Reversing', 'Others'])
casualty_class = st.selectbox('Casualty Class', options=['Driver or rider', 'Passenger', 'Pedestrian'])
sex_of_casualty = st.selectbox('Sex of Casualty', options=['Male', 'Female'])
age_band_of_casualty = st.selectbox('Age Band of Casualty', options=['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '91-95', '96-100', 'Unknown'])
casualty_severity = st.selectbox('Casualty Severity', options=['Slight Injury', 'Serious Injury', 'Fatal Injury'])
work_of_casuality = st.selectbox('Work of Casualty', options=['Driver', 'Passenger', 'Pedestrian', 'Unknown'])
fitness_of_casuality = st.selectbox('Fitness of Casualty', options=['Normal', 'Deaf', 'Blind', 'Other'])
pedestrian_movement = st.selectbox('Pedestrian Movement', options=['Crossing', 'Standing', 'Moving', 'Not a Pedestrian'])
cause_of_accident = st.selectbox('Cause of Accident', options=['Overspeed', 'Overtaking', 'Changing lane', 'Other'])
accident_severity = st.selectbox('Accident_severity',options=['Slight Injury','Serious Injury'])

# Create a dictionary of inputs
input_data = {
    'Time': time,
    'Day_of_week': day_of_week,
    'Age_band_of_driver': age_band_of_driver,
    'Sex_of_driver': sex_of_driver,
    'Educational_level': educational_level,
    'Vehicle_driver_relation': vehicle_driver_relation,
    'Driving_experience': driving_experience,
    'Type_of_vehicle': type_of_vehicle,
    'Owner_of_vehicle': owner_of_vehicle,
    'Service_year_of_vehicle': service_year_of_vehicle,
    'Defect_of_vehicle': defect_of_vehicle,
    'Area_accident_occured': area_accident_occured,
    'Lanes_or_Medians': lanes_or_medians,
    'Road_allignment': road_allignment,
    'Types_of_Junction': types_of_junction,
    'Road_surface_type': road_surface_type,
    'Road_surface_conditions': road_surface_conditions,
    'Light_conditions': light_conditions,
    'Weather_conditions': weather_conditions,
    'Type_of_collision': type_of_collision,
    'Number_of_vehicles_involved': number_of_vehicles_involved,
    'Vehicle_movement': vehicle_movement,
    'Casualty_class': casualty_class,
    'Sex_of_casualty': sex_of_casualty,
    'Age_band_of_casualty': age_band_of_casualty,
    'Casualty_severity': casualty_severity,
    'Work_of_casuality': work_of_casuality,
    'Fitness_of_casuality': fitness_of_casuality,
    'Pedestrian_movement': pedestrian_movement,
    'Cause_of_accident': cause_of_accident,
    'Accident_severity':accident_severity
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Make predictions
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f'Predicted Number of Casualties: {prediction[0]:.2f}')
