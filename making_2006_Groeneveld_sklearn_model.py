# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:16:14 2025

@author: User
"""

import numpy as np
import pandas as pd
import joblib


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

skip = False

if skip == False:
    # Load and preprocess data
    df = pd.read_csv(r"C:\Users\User\Documents\ThermalHydraulics\SCAcode_v0\2006_Groeneveld_LUT.csv")
    df.drop(index=df.index[0], axis=0, inplace=True)  # Remove the first row (likely headers)
    df.dropna(inplace=True)  # Remove rows with missing values
    
    # Extract input features and target
    X = np.column_stack([
        df["D"].to_numpy(dtype=float),       # Hydraulic diameter (m)
        df["L"].to_numpy(dtype=float),       # Heated length (m)
        df["P"].to_numpy(dtype=float),       # Pressure (kPa)
        df["G"].to_numpy(dtype=float),       # Mass flux (kg/m^2*s)
        df["Xchf"].to_numpy(dtype=float),    # Quality (unitless)
        df["DHin"].to_numpy(dtype=float),    # Inlet subcooling (kJ/kg)
        df["Tin"].to_numpy(dtype=float)      # Inlet temperature (°C)
    ])
    
    y = df["CHF"].to_numpy(dtype=float)  # Critical Heat Flux (kW)
    
    
    
    # Train regression model
    #model = make_pipeline(StandardScaler(), SVR(C=1000, epsilon=0.1)) 757 got 865
    #model = DecisionTreeRegressor(random_state=0) # will act like a LUT, is good and will be conservative, but doesnt interp
    model = RandomForestRegressor(n_estimators=100, random_state=42) # and relative fast pretty good at 772, 1003 and 856
    #model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, random_state=42) # not very good
    #model = XGBRegressor(n_estimators=10000, learning_rate=0.25, random_state=42) # ehhhhhhhhhh better but not that good
    
    model.fit(X, y)
    
    # Save the trained model
    model_filename = 'groeneveld_chf_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")


model_filename = "groeneveld_chf_model.pkl"
# Example of reloading and using the model
loaded_model = joblib.load(model_filename)

# Example query point (make sure it’s 2D)
query_point1 = np.array([[0.004, 0.396, 100, 142.7, 0.79, 317, 23.94]]) # 757
query_point2 = np.array([[0.004, 0.396, 100, 203.6, 0.7, 317, 23.94]]) # 978
query_point3 = np.array([[0.004, 0.396, 100, 173.3, 0.745, 317, 23.94]]) # 867.5


predicted_chf1 = loaded_model.predict(query_point1)
predicted_chf2 = loaded_model.predict(query_point2)
predicted_chf3 = loaded_model.predict(query_point3)