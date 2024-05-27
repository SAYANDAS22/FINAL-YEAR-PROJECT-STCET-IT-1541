import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# importing the dataset
df = pd.read_csv('float_merged_updated1.csv')
st.set_page_config(
    page_title="Cyclone Predictor",
    layout="centered"
)
st.title("Cyclone Casualty and Property Loss Predictor")


states = df["STATE"].unique().tolist()


def all_predictions(state, wind_speed):
    precautions = "Stay indoors and away from windows until the storm has passed. Take refuge in an interior room, closet or hallway on the lowest floor possible. Don't use electrical equipment or landline phones during the storm due to lightning risk. Avoid walking or driving through floodwaters. Clear loose objects from around your property that could cause damage in high winds. Stock up on emergency supplies like non-perishable foods, water, medications, batteries, and first aid kit."
    # Training 80% and Testing 20%
    state_df = df[df['STATE'] == state][['WIND SPEED', 'property loss', 'CASUALITIES']]
    X = state_df[['WIND SPEED']]

    # Training and Testing the model for Proerty loss and Its Prediction
    y = np.array(state_df[['property loss', 'CASUALITIES']])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)

    def prediction(wind_speed):
        l = rf.predict([[wind_speed]])[0].tolist()
        l.append(precautions)
        return l
    

    return prediction(wind_speed)

state = st.selectbox(label="Select the State", options=states)

wind = st.number_input(label="Enter the Wind Speed",placeholder="in Km/hr",step=1, min_value=85,max_value=250)

# c1,c2,c3 = st.columns([5,6,1])
# with c2:

if  st.button("Predict"):
    if state and wind: 
        prediction = all_predictions(state, wind)
        st.write(f"Estimated Property Loss is : {round(prediction[0]*100,3)}%")
        st.write(f"Estimated Casualties are : {round(prediction[1]*100,3)}%")
        st.write(f"Precautions to be taken : {prediction[2]}")



        