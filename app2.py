import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ===================================================================
# ECO-HEALTH-VIZ: ENVIRONMENTAL AND INFECTIOUS DISEASE EXPLORER
# ===================================================================
# This Streamlit application explores simulated county-level records.
# Comments explain the interface, data generation, and chart workflow.
# All observations are synthetic, including the county names.

# 1. PAGE CONFIGURATION
# Use a wide layout; the sidebar adapts to the device screen size.
st.set_page_config(
    page_title="Eco-Health-Viz: Climate & Infectious Disease Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "Eco-Health-Viz, developed by Mohammad Shaito, Ph.D. A synthetic-data demonstration of environmental and infectious disease visualization."}
)

# ===================================================================
# 2. APPLICATION HEADER, ABOUT, AND USER INSTRUCTIONS
# ===================================================================
st.markdown("# 🌍 Eco-Health-Viz: Environmental & Public Health Dashboard")
st.markdown("#### *(Spatiotemporal exploration of environmental stressors and infectious disease dynamics.)*")
st.markdown("👨‍💻 **Developed by [Mohammad Shaito, Ph.D.](https://sites.google.com/view/mohammad-shaito/home)**")

# Expanders keep supporting information available without crowding charts.
with st.expander("About Eco-Health-Viz"):
    st.markdown("""
### Welcome to Eco-Health-Viz
Explore simulated relationships among temperature anomalies, fine particulate
matter (PM2.5), household income, and COVID-19 and influenza case rates.

**Data coverage:** 50 fictional counties per selected state, with monthly
observations from January 2021 through December 2023.

**Data source:** All records are generated within this application using a
fixed random seed. This version does not load Johns Hopkins, environmental,
or other surveillance datasets. Simulated associations are built into the
generator and do not establish real-world relationships or causation.

**Developer:** [Mohammad Shaito, Ph.D.](https://sites.google.com/view/mohammad-shaito/home)

**Contact:** [mohammad.shaito@mavs.uta.edu](mailto:mohammad.shaito@mavs.uta.edu)
or [mshaito78@gmail.com](mailto:mshaito78@gmail.com).
""")

with st.expander("Show Instructions"):
    st.markdown("""
1. **Select a state** in the sidebar.
2. **Choose an analysis mode** to display a scatter plot, temporal comparison,
   or simulated data table.
3. **Select a start and end date.** Charts and the CSV download use this range.
   Observations occur at month end, so short intervals may contain no records.
4. In **Environmental Stress vs. Disease Incidence**, select the environmental
   or economic variable and epidemiological metric.
5. In **Multi-Pathogen & Climate Comparison**, compare monthly county-average
   case rates with PM2.5 on a separate axis.
6. In **Synthetic Simulation Mode**, inspect up to 100 rows and download all
   filtered simulated records as CSV.

Hover over charts to inspect values. Use chart controls to zoom or reset.
""")

with st.expander("📱 Access on a Mobile Device"):
    st.markdown("""
Open this app's deployed web address in your phone or tablet browser.
Tap the arrow or sidebar icon at the top-left to open the controls, make your
selections, and close the sidebar to see the charts. Rotate to landscape for
more chart space. The Python file must first be running on a Streamlit server.
""")

# Show the short sidebar reminder only on screens up to 768 pixels wide.
# This CSS changes presentation only; it does not affect calculations.
st.markdown(
    """
    <style>
        .mobile-tip { display: none; }
        @media (max-width: 768px) {
            .mobile-tip {
                display: block;
                margin: 8px 0 12px;
                padding: 8px 10px;
                font-size: 0.90rem;
                line-height: 1.4;
            }
        }
    </style>
    <div class="mobile-tip">
        📱 Tap the arrow or sidebar icon at the top-left to open the controls.
    </div>
    """,
    unsafe_allow_html=True,
)
st.info("Demonstration data: all counties and observations shown here are simulated.")

# 3. SIDEBAR CONTROLS
# Streamlit reruns the script when a selection changes.
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
    value=(pd.to_datetime("2021-01-01").date(), pd.to_datetime("2023-12-31").date()),
    min_value=pd.to_datetime("2021-01-01").date(),
    max_value=pd.to_datetime("2023-12-31").date()
)

# 4. SYNTHETIC DATA GENERATION
# Cache each state dataset so widget changes do not regenerate its records.
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

# 5. LOAD AND FILTER DATA
# A date-range widget may temporarily contain just one endpoint.
# Wait until both endpoints are selected before rendering the dashboard.
if len(date_range) != 2:
    st.info("Select both a start date and an end date in the sidebar.")
    st.stop()

start_date, end_date = map(pd.Timestamp, date_range)
df = generate_synthetic_eco_health_data(selected_state)
df = df.loc[df["Date"].between(start_date, end_date)].copy()
if df.empty:
    st.warning("No monthly observations fall within this range. Include a month-end date.")
    st.stop()
st.caption(f"{selected_state} | {start_date:%b %d, %Y} to {end_date:%b %d, %Y} | {len(df):,} simulated records")

# 6. DISPLAY THE SELECTED ANALYSIS
# Each mode uses the same state and date-filtered dataset.
if analysis_mode == "Environmental Stress vs. Disease Incidence":
    st.subheader(f"Simulated Socioeconomic, Environmental & Disease Patterns ({selected_state})")
    
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
    st.subheader("Time-Series Trends: Simulated Case Rates vs. Air Quality (PM2.5)")
    
    # Average county rates for each month. These are unweighted means,
    # not population-weighted statewide incidence estimates.
    trend_df = df.groupby("Date")[["COVID_Per_Capita", "Flu_Per_Capita", "PM25"]].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["COVID_Per_Capita"], name="COVID-19 Cases / 100k", yaxis="y"))
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Flu_Per_Capita"], name="Flu Cases / 100k", yaxis="y"))
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["PM25"], name="PM2.5 (µg/m³)", yaxis="y2", line=dict(dash="dash", color="orange")))
    
    fig.update_layout(
        title="Synchronized Temporal Trends: Respiratory Illnesses & Air Quality",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Disease Cases per 100k inhabitants"),
        yaxis2=dict(title="PM2.5 (µg/m³)", overlaying="y", side="right"),
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("Synthetic Data Inspection & Validation Explorer")
    st.markdown("Inspect the filtered synthetic records used in the visualizations.")
    st.dataframe(df.head(100), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Simulated Eco-Health Dataset (CSV)",
        data=csv,
        file_name=f"eco_health_data_{selected_state.lower()}.csv",
        mime='text/csv',
    )

# Developer attribution is also available in the About section above.
st.caption("Eco-Health-Viz | Developed by Mohammad Shaito, Ph.D. | Synthetic-data demonstration")
