import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Eco-Health-Viz: Climate & Infectious Disease Dashboard",
    page_icon="🌍",
    layout="wide"
)

# App Header
st.title("Eco-Health-Viz: Spatiotemporal Analysis of Environmental Stressors and Infectious Disease Dynamics")
st.markdown("""
An interactive web-based dashboard for exploring county-level intersections between 
environmental indicators (temperature anomalies, $\mathrm{PM}_{2.5}$) and multi-pathogen epidemiological trends.
""")

# Sidebar Controls
st.sidebar.header("Configuration & Filtering")

# State Selection (focused on Texas or nationwide expansion)
selected_state = st.sidebar.selectbox("Select State", ["Texas", "California", "New York", "Florida", "Illinois"])

# Analysis Mode
analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode", 
    ["Environmental Stress vs. Disease Incidence", "Multi-Pathogen & Climate Comparison", "Synthetic Simulation Mode"]
)

# Date Range Simulation Filter (Corrected to use st.date_input)
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(pd.to_datetime("2021-01-01").date(), pd.to_datetime("2023-12-31").date())
)

# Synthetic Data Generation Engine (Fallback & Prototyping)
@st.cache_data
def generate_synthetic_eco_health_data(state_name):
    """Generates synthetic county-level environmental and epidemiological records."""
    np.random.seed(42)
    counties = [f"{state_name} County {i+1}" for i in range(50)]
    dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="ME")
    
    data = []
    for date in dates:
        for county in counties:
            # Simulate environmental stressors
            temp_anomaly = np.random.normal(loc=1.5, scale=0.8) # Celsius offset
            pm25 = np.random.gamma(shape=2.0, scale=4.0) # µg/m³
            
            # Simulate population & income
            population = np.random.randint(20000, 1500000)
            median_income = np.random.normal(loc=65000, scale=15000)
            
            # Disease rates influenced slightly by simulated environmental stress
            covid_cases = max(0, int(np.random.poisson(lam=50) + (pm25 * 2) + (temp_anomaly * 5)))
            flu_cases = max(0, int(np.random.poisson(lam=30) + (pm25 * 1.5)))
            
            covid_per_capita = (covid_cases / population) * 100000
            flu_per_capita = (flu_cases / population) * 100000
            
            data.append({
                "Date": date,
                "State": state_name,
                "County": county,
                "Population": population,
                "Median_Household_Income": round(median_income, 2),
                "Temperature_Anomaly": round(temp_anomaly, 2),
                "PM25": round(pm25, 2),
                "COVID_Cases": covid_cases,
                "Flu_Cases": flu_cases,
                "COVID_Per_Capita": round(covid_per_capita, 2),
                "Flu_Per_Capita": round(flu_per_capita, 2)
            })
            
    return pd.DataFrame(data)

# Load Data
df = generate_synthetic_eco_health_data(selected_state)

# Main Dashboard Logic
if analysis_mode == "Environmental Stress vs. Disease Incidence":
    st.subheader(f"Socioeconomic & Environmental Impact on COVID-19 Incidence ({selected_state})")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis_val = st.selectbox("Select Environmental / Economic Variable", ["PM25", "Temperature_Anomaly", "Median_Household_Income"])
    with col2:
        y_axis_val = st.selectbox("Select Epidemiological Metric", ["COVID_Per_Capita", "Flu_Per_Capita"])
        
    fig = px.scatter(
        df, 
        x=x_axis_val, 
        y=y_axis_val, 
        color="County",
        size="Population",
        hover_name="County",
        title=f"{y_axis_val} vs. {x_axis_val} Across Counties",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "Multi-Pathogen & Climate Comparison":
    st.subheader("Time-Series Trends: Pathogen Prevalence vs. Air Quality ($\mathrm{PM}_{2.5}$)")
    
    # Aggregate data monthly
    trend_df = df.groupby("Date")[["COVID_Per_Capita", "Flu_Per_Capita", "PM25"]].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["COVID_Per_Capita"], name="COVID-19 Cases / 100k", yaxis="y"))
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Flu_Per_Capita"], name="Flu Cases / 100k", yaxis="y"))
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["PM25"], name="PM2.5 Index (µg/m³)", yaxis="y2", line=dict(dash="dash", color="orange")))
    
    fig.update_layout(
        title="Synchronized Temporal Trends: Respiratory Illnesses & Air Quality",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Disease Cases per 100k inhabitants"),
        yaxis2=dict(title="PM2.5 Level", overlaying="y", side="right"),
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("Synthetic Data Inspection & Validation Explorer")
    st.markdown("Inspect raw harmonized records generated by the ingestion engine before visualization mapping.")
    st.dataframe(df.head(100), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Simulated Eco-Health Dataset (CSV)",
        data=csv,
        file_name=f"eco_health_data_{selected_state.lower()}.csv",
        mime='text/csv',
    )
