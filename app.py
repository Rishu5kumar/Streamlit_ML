import streamlit as st
import pandas as pd
import numpy as np
import joblib
import genai
import matplotlib.pyplot as plt
import re
import html
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to preprocess input data
def preprocess_input(average_temp, year, month, country, encoder):
    features = np.array([[average_temp, year, month]])
    country_encoded = encoder.transform([[country]])
    input_data = np.concatenate((features, country_encoded), axis=1)
    return input_data

# Function to generate descriptive paragraph using genai
def generate_paragraph(avg_temp, predicted_uncertainty, year, month, country):
    month_name = pd.to_datetime(f"{year}-{month}-01").strftime("%B")
    
    prompt = (
        f"In {country}, the average temperature recorded for {month_name} {year} was {avg_temp:.2f}°C. "
        f"According to the model's predictions, the uncertainty in this temperature is estimated to be approximately {predicted_uncertainty:.2f}°C. "
        f"This level of uncertainty could significantly impact various sectors, including agriculture, energy consumption, and general climate conditions in {country}. "
        f"To mitigate these impacts, please provide preventive measures that can be taken in the following areas: "
        f"1. Agriculture: Consider strategies to adapt to changing temperatures and manage crop production. "
        f"2. Energy Consumption: Explore ways to enhance energy efficiency and manage energy use. "
        f"3. Climate Conditions: Identify steps to improve resilience to potential climate impacts. "
        f"Understanding and addressing these uncertainties is crucial for effective planning and decision-making in these areas."
    )

    # Use API key directly in method call or through environment variable
    api_key = st.secrets["AIzaSyCX3I28pHzmiSEM6Rt1kdVX7e2BhwSuOOA"]  # Ensure this secret is added in Streamlit Cloud
    response = genai.generate_text(prompt, api_key=api_key)
    return format_response(response)

# Function to format the response
def format_response(text):
    formatted_text = html.escape(text)
    formatted_text = formatted_text.strip()
    formatted_text = formatted_text.replace('\n', '<br>')
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_text)
    return formatted_text

# Streamlit app
st.title('Temperature Uncertainty Predictor')

st.sidebar.header('Input Data')
average_temp = st.sidebar.number_input('Average Temperature (°C)', value=27.5, format="%.1f")
year = st.sidebar.number_input('Year', value=2024, min_value=1900, max_value=2100)
month = st.sidebar.number_input('Month', value=7, min_value=1, max_value=12)
country = st.sidebar.text_input('Country', value='India')

# Load the trained model and encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

# Predict button
if st.sidebar.button('Predict'):
    input_data = preprocess_input(average_temp, year, month, country, encoder)
    predicted_uncertainty = model.predict(input_data)[0]
    st.write(f"**Predicted Temperature Uncertainty** for {country} in {month}/{year} with avg_temp {average_temp}°C: {predicted_uncertainty:.2f}°C")
    
    # Generate paragraph
    description = generate_paragraph(average_temp, predicted_uncertainty, year, month, country)
    st.write(description, unsafe_allow_html=True)

# Load the historical data for plotting
df = pd.read_csv('temperature.csv')
df['dt'] = pd.to_datetime(df['dt'])
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month

# Ensure all expected columns are in the DataFrame
expected_columns = encoder.get_feature_names_out(['Country'])
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value=0)

# Apply imputers
df[['AverageTemperature', 'year', 'month']] = num_imputer.fit_transform(df[['AverageTemperature', 'year', 'month']])
df[expected_columns] = cat_imputer.fit_transform(df[expected_columns])

# Prepare data for model evaluation
X = df[['AverageTemperature', 'year', 'month'] + list(expected_columns)]
y = df['AverageTemperatureUncertainty']

# Check and remove any remaining NaN values
if np.any(pd.isna(X)):
    st.write("Warning: Some input data contains NaN values, which have been imputed.")
    X = np.nan_to_num(X)

y_pred = model.predict(X)

# Model evaluation metrics
st.sidebar.header('Model Evaluation Metrics')
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

st.sidebar.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.sidebar.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.sidebar.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.sidebar.write(f"**R² Score:** {r2:.2f}")

# Plotting the results
st.header('Model Performance')

results_df = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred
})

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results_df.index, results_df['Actual'], label='Actual Values', marker='o', linestyle='-')
ax.plot(results_df.index, results_df['Predicted'], label='Predicted Values', marker='x', linestyle='--')
ax.set_xlabel('Index')
ax.set_ylabel('Average Temperature')
ax.set_title('Actual vs. Predicted Average Temperature')
ax.legend()
ax.grid(True)

st.pyplot(fig)
