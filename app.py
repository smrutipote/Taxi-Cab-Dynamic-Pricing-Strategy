import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gradio as gr

# Load and preprocess data
df = pd.read_csv("dynamic_pricing.csv")

# Convert Vehicle_Type to numeric
df['Vehicle_Type'] = np.where(df['Vehicle_Type'] == 'Premium', 1, 0)

# Calculate demand_multiplier
df['demand_multiplier'] = np.where(
    df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], 75),
    df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], 75),
    df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], 25)
)

# Calculate supply_multiplier
df['supply_multiplier'] = np.where(
    df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], 25),
    df['Number_of_Drivers'] / np.percentile(df['Number_of_Drivers'], 75),
    df['Number_of_Drivers'] / np.percentile(df['Number_of_Drivers'], 25)
)

# Compute adjusted ride cost
df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (
    np.maximum(df['demand_multiplier'], 0.8) *
    np.maximum(df['supply_multiplier'], 0.8)
)

# Model training
X = df[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]]
y = df["adjusted_ride_cost"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Helper function
def get_numeric_vehicle_type(vehicle_type):
    return {'Premium': 1, 'Economy': 0}.get(vehicle_type, 0)

# Prediction function
def predict_price(number_of_riders, number_of_drivers, vehicle_type, expected_duration):
    vehicle_type_numeric = get_numeric_vehicle_type(vehicle_type)
    inputs = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, expected_duration]])
    prediction = model.predict(inputs)[0]
    return round(prediction, 2)

# Gradio UI
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 100, value=50, label="Number of Riders"),
        gr.Slider(0, 100, value=25, label="Number of Drivers"),
        gr.Radio(["Premium", "Economy"], label="Vehicle Type"),
        gr.Slider(1, 120, value=30, label="Expected Ride Duration (minutes)")
    ],
    outputs=gr.Number(label="Predicted Ride Cost (€)"),
    title="Dynamic Pricing Predictor",
    description="Estimate the adjusted ride cost based on current demand, supply, and ride details."
)

demo.launch()