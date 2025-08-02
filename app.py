import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import os

df = pd.read_csv("household_energy.csv")

st.title("Household Energy Dashboard")


usage_levels = ["All"] + sorted(df["device_usage"].unique().astype(str).tolist())
selected_usage = st.sidebar.selectbox("Select Device Usage", usage_levels)

if selected_usage != "All":
    df = df[df["device_usage"].astype(str) == selected_usage]

st.subheader("Energy consumption overview")
st.dataframe(df, use_container_width=True)

avg_energy = df["energy_consumption"].mean()
total_energy = df["energy_consumption"].sum()
st.metric("Average Energy Consumption (kWh)", f"{avg_energy:.2f}")
st.metric("Total Energy Consumption (kWh)", f"{total_energy:.2f}")

st.subheader("Predict Energy Consumption")

model_path = os.path.join("energy_app", "forecast_model.pkl")

if not os.path.exists(model_path):
    st.error(f"Could NOT find the model at: {model_path}. Please check your folder and filename.")
    st.stop()

model = joblib.load(model_path)

temp = st.slider("Indoor Temperature (°C)", 10.0, 40.0, 25.0)
outside_temp = st.slider("Outside Temperature (°C)", 10.0, 45.0, 30.0)
device_usage = st.slider("Active Devices", 0, 10, 2)
hour = st.slider("Hour of Day", 0, 23, 14)
weekday = st.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)))


input_data = pd.DataFrame([[temp, outside_temp, device_usage, hour,weekday]],
                          columns=["temperature", "outside_temperature", "device_usage", "hour",'weekday'])

prediction = model.predict(input_data)[0]
st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")

st.subheader("Outside Temperature vs Energy Consumption")
fig1, ax1 = plt.subplots()
sns.scatterplot(
    data=df,
    x="outside_temperature",
    y="energy_consumption",
    ax=ax1,
)
ax1.set_xlabel("Outside Temperature (°C)")
ax1.set_ylabel("Energy Consumption (kWh)")
st.pyplot(fig1)

st.subheader("Device Usage vs Energy Consumption")
fig2, ax2 = plt.subplots()

avg_by_usage = df.groupby("device_usage")["energy_consumption"].mean().reset_index()
sns.barplot(data=avg_by_usage, x="device_usage", y="energy_consumption", ax=ax2)
ax2.set_xlabel("Device Usage (number of active devices)")
ax2.set_ylabel("Avg Energy Consumption (kWh)")
st.pyplot(fig2)

st.subheader("Smart Recommendations")
threshold = df["energy_consumption"].mean() * 1.5 
for _, row in df.iterrows():
    if row["energy_consumption"] > threshold:
        st.warning(
            f"Timestamp {row['timestamp']} - High usage ({row['energy_consumption']:.2f} kWh). Recommend checking appliances or switching to more efficient options."
        )
    elif row["device_usage"] >= 1:
        st.info(
            f"Timestamp {row['timestamp']} - Device usage is {row['device_usage']}. Consider sub-metering active devices for better billing insight."
        )