import streamlit as st
import datetime
import time
import pandas as pd
import numpy as np
import tensorflow as tf

# Function to calculate energy based on the selected date
# Function to load the pre-trained model

def generate_features(date):
    features = {
        'dayofyear': date.dayofyear,
        'hour': date.hour,
        'dayofweek': date.dayofweek,
        'quarter': (date.month - 1) // 3 + 1,
        'month': date.month,
        'year': date.year
    }
    return features

# Function to make predictions using the TensorFlow model
def predict_energy_export(model, input_sequence, scaler_X, scaler_y):
    # Convert input_sequence to a NumPy array
    input_data = np.array([list(feature.values()) for feature in input_sequence])

    # Reshape input_data to match the model's input shape
    input_data = input_data.reshape((1, SEQUENCE_LENGTH, len(FEATURES)))

    # Make predictions
    input_data = scaler_X.transform(input_data)
    predictions = model.predict(input_data)
    return scaler_y.inverse_transform(predictions[0][0])  # Assuming a single output neuron for regression

# Function to load the TensorFlow model
def load_tf_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to get the last day of the month for a given date
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + timedelta(days=4)  # go to the 28th, then forward 4 days
    return next_month - timedelta(days=next_month.day)

# Streamlit app
def main():
    st.title("Energy Calculator App ⚡🍃")

    # Image
    # st.image("wind-turbine.jpg", caption="Wind Turbine", use_column_width=True)
    # Input date for prediction
    input_date_str = input("Enter the date (format: YYYY-MM-DD HH:mm): ")
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d %H:%M')

    # Load the TensorFlow model
    model_path = 'modelTS.h5'
    model = load_tf_model(model_path)

    scaler_X_path = 'scaler_X.joblib'
    scaler_y_path = 'scaler_y.joblib'

    # Load the MinMaxScaler
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Initialize input_sequence with default values
    input_sequence = [generate_features(input_date - timedelta(minutes=i * 10)) for i in range(SEQUENCE_LENGTH)]

    # Generate predictions until the end of the month
    last_day = last_day_of_month(input_date)
    current_date = input_date

    prediction = predict_energy_export(model, input_sequence, scaler_X, scaler_y)

    chart_data = pd.DataFrame({
        "Time": days,
        "Energy Produced in KWH": prediction,
    })

    # Calculate energy button
    if st.button("Calculate Energy"):
        # Display loading bar while calculating energy
        with st.spinner("Calculating energy..."):
            # Call the function to calculate energy
            result = calculate_energy(selected_date)
            # Simulate delay for loading bar effect
            time.sleep(1)

            # Display the result
            st.success(result)

    st.line_chart(chart_data, x="Time", y="Energy Produced in KWH")
if __name__ == "__main__":
    main()
