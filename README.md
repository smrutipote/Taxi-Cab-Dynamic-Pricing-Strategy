# Taxi-Service-Dynamic-Pricing-Strategy
Read the complete code explaination :https://medium.com/@smruti.po1106/taxi-service-dynamic-pricing-strategy-using-machine-learning-6213728d0c8e

Also try it here : https://huggingface.co/spaces/smrup/Taxi_Dynamic_Pricing_Strategy

This project demonstrates how to build a Dynamic Pricing Strategy for ride-sharing services using Python, data science techniques, and machine learning. The goal is to dynamically adjust ride prices based on demand and supply factors to optimize revenue and customer experience.

📌 Project Overview
The traditional pricing model used by the company is based solely on the expected ride duration, which fails to account for real-time market dynamics. This project introduces a data-driven approach using demand/supply multipliers and a Random Forest Regression model to estimate adjusted ride costs more effectively.

📊 Dataset
The dataset dynamic_pricing.csv contains:

Number of Riders

Number of Drivers

Expected Ride Duration

Historical Cost of Ride

Vehicle Type

Location Category

Average Ratings

Number of Past Rides

🧪 Key Features
🔍 Data Analysis & Visualization
Distribution of ride durations and cost

Box plots by vehicle type

Correlation matrix heatmap

📈 Dynamic Pricing Logic
Demand Multiplier: Adjusted based on percentiles of rider demand

Supply Multiplier: Adjusted based on percentiles of driver availability

Price Adjustment Formula:
adjusted_ride_cost = historical_cost * max(demand_multiplier, threshold) * max(supply_multiplier, threshold)
💰 Profitability Analysis
Calculates profit percentage post dynamic adjustment

Visualized using a donut chart comparing profitable vs loss-making rides

🔄 Preprocessing Pipeline
Handles missing values and outliers

Converts categorical values to numeric

🤖 Machine Learning Model
Trains a Random Forest Regressor to predict adjusted ride cost

Input features: Number of Riders, Number of Drivers, Vehicle Type, Expected Ride Duration

🧠 Prediction Function
A function predict_price() is provided to input custom values and get a predicted dynamic price:

predict_price(
    number_of_rides=50,
    number_of_drivers=25,
    vehicle_type="Economy",
    Expected_Ride_Duration=30

