import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import streamlit as st

# Page config
st.set_page_config(page_title="Interactive Sales Forecasting", layout="wide")

# Header
st.title(" Interactive Sales Forecasting Dashboard")
st.markdown("Upload your data or use sample data. Choose forecast period and method to see results.")

# Sidebar
st.sidebar.header("Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

# Use uploaded data or fallback to default
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df = df.sort_values(by='Date')
else:
    st.sidebar.info("Using sample data (Jan–Dec 2023).")
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='ME'),
        'Sales': [200, 220, 250, 275, 300, 320, 340, 360, 400, 420, 450, 480]
    })

df['Month'] = range(1, len(df) + 1)

# Forecast controls
forecast_months = st.sidebar.slider("Months to Forecast", 1, 12, 6)
model_type = st.sidebar.selectbox("Forecast Model", ["Linear Regression", "Polynomial Regression (Degree 2)"])

# Filter by date
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input("Filter Date Range", [min_date, max_date])
if len(date_range) == 2:
    df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

# Train model
X = df[['Month']]
y = df['Sales']
future_months = pd.DataFrame({'Month': range(len(df)+1, len(df)+forecast_months+1)})

if model_type == "Polynomial Regression (Degree 2)":
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    future_poly = poly.transform(future_months)
    model = LinearRegression()
    model.fit(X_poly, y)
    future_sales = model.predict(future_poly)
else:
    model = LinearRegression()
    model.fit(X, y)
    future_sales = model.predict(future_months)

future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='ME')
forecast_df = pd.DataFrame({'Date': future_dates, 'Sales': future_sales})

# Combine data
full_df = pd.concat([df[['Date', 'Sales']], forecast_df], ignore_index=True)

# Plot: Forecast Chart
st.subheader("Forecast Chart")
fig = px.line(full_df, x='Date', y='Sales', markers=True, title="Sales Forecast")
fig.add_scatter(x=df['Date'], y=df['Sales'], mode='lines+markers', name='Actual')
fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Sales'], mode='lines+markers', name='Forecast')
st.plotly_chart(fig, use_container_width=True)

# Growth chart
df['Growth %'] = df['Sales'].pct_change() * 100
st.subheader("Monthly Sales Growth")
fig2 = px.bar(df, x='Date', y='Growth %', title="Monthly Growth (%)", text_auto='.2f')
st.plotly_chart(fig2, use_container_width=True)

# Data Table
with st.expander("View Full Data Table"):
    st.dataframe(full_df, use_container_width=True)

# Download forecast data
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecasted Data as CSV",
    data=csv,
    file_name='forecasted_sales.csv',
    mime='text/csv',
)

