"""Eco-Health-Viz 3.0: transparent synthetic scenarios, not observed surveillance.

The pure functions above main() are independently importable and testable.
Run with: streamlit run app.py
"""
from pathlib import Path
import hashlib
import json
import platform
import re
from importlib.metadata import version
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# This version identifies the simulation rules in the interface and exports.
GENERATOR_VERSION = "3.0.0"
# All values below are arbitrary scenario settings, not state estimates.
SCENARIOS = {
    "Texas": {"seed": 4101, "pm": 8.0, "temp": 1.0, "covid": 100.0, "flu": 60.0},
    "California": {"seed": 4102, "pm": 9.0, "temp": 1.2, "covid": 120.0, "flu": 55.0},
    "New York": {"seed": 4103, "pm": 7.0, "temp": 0.8, "covid": 90.0, "flu": 65.0},
    "Florida": {"seed": 4104, "pm": 6.0, "temp": 1.4, "covid": 110.0, "flu": 50.0},
    "Illinois": {"seed": 4105, "pm": 8.5, "temp": 0.9, "covid": 105.0, "flu": 70.0},
}
# These labels connect the analysis menu to the three display branches in main().
MODES = ["Environmental and disease comparison", "Monthly trends", "Data inspection and export"]
X_VARIABLES = ["PM25", "Temperature_Anomaly", "Median_Household_Income"]
Y_VARIABLES = ["COVID_Per_Capita", "Flu_Per_Capita"]
# Map internal column names to chart labels with units.
LABELS = {"PM25": "Simulated PM2.5 (µg/m³)",
          "Temperature_Anomaly": "Simulated temperature offset (°C)",
          "Median_Household_Income": "Synthetic household income (nominal units)",
          "COVID_Per_Capita": "Simulated COVID-19 cases per 100,000 per month",
          "Flu_Per_Capita": "Simulated influenza cases per 100,000 per month"}


def generate_synthetic_eco_health_data(state_name):
    """Create fictional county records using seasonality and Poisson counts.

    Uniform county attributes are fixed across months. Independent normal noise
    is added to environmental inputs. Disease expectations do not use environmental
    inputs or income. Parameters are illustrative, not fitted to observations.
    """
    # Separate streams keep case sampling independent of environmental draws.
    cfg = SCENARIOS[state_name]
    env_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0]))
    count_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1]))
    # Use one row per county per month. ME selects the last day of each month.
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="ME")
    rows = []
    for i in range(50):
        # Draw county attributes once so they stay fixed over time.
        population = int(env_rng.integers(20000, 500001))
        income = round(float(env_rng.uniform(35000, 100000)), 2)
        risk = float(env_rng.uniform(0.8, 1.2))
        pm_base = float(cfg["pm"] * env_rng.uniform(0.8, 1.2))
        for t, date in enumerate(dates):
            # Share a February-peaking annual cycle across the synthetic series.
            season = float(np.cos(2 * np.pi * (t - 1) / 12))
            # Add random variation to each environmental value for this month.
            pm_error = float(env_rng.normal(0, 0.5))
            temp_error = float(env_rng.normal(0, 0.12))
            pm = max(0.1, pm_base * (1 + 0.15 * season) + pm_error)
            temp = cfg["temp"] + 0.2 * season + temp_error
            # Disease rates use county and seasonal factors. PM2.5 and temperature are comparison variables.
            covid_rate = cfg["covid"] * risk * (1 + 0.25 * season)
            flu_rate = cfg["flu"] * risk * (1 + 0.65 * season)
            # Convert rates per 100,000 to expected counts before sampling.
            covid_mu = population * covid_rate / 100000
            flu_mu = population * flu_rate / 100000
            # Poisson counts: conditional mean and variance both equal mu.
            covid = int(count_rng.poisson(covid_mu))
            flu = int(count_rng.poisson(flu_mu))
            # Keep sampled counts and expected values together so users can inspect both in the export.
            rows.append({"Date": date, "State": state_name,
                         "County": f"{state_name} Synthetic County {i + 1:02d}",
                         "Population": population, "Median_Household_Income": income,
                         "Temperature_Anomaly": temp, "PM25": pm,
                         "COVID_Cases": covid, "Flu_Cases": flu,
                         "COVID_Per_Capita": covid / population * 100000,
                         "Flu_Per_Capita": flu / population * 100000,
                         "Expected_COVID_Rate": float(covid_rate), "Expected_Flu_Rate": float(flu_rate),
                         "Expected_COVID_Count": float(covid_mu), "Expected_Flu_Count": float(flu_mu),
                         "County_Multiplier": risk, "County_PM_Baseline": pm_base,
                         "Seasonal_Cosine": season, "Generator_Version": GENERATOR_VERSION,
                         "Scenario_Seed": cfg["seed"]})
    # Convert the list of records to a table and put rows in date and county order.
    return pd.DataFrame(rows).sort_values(["Date", "County"]).reset_index(drop=True)


def filter_records(df, date_range, counties=None):
    """Apply inclusive dates and an optional county subset; reject partial ranges."""
    # The date widget can temporarily return one endpoint while the user picks the other.
    if len(date_range) != 2:
        raise ValueError("Select both a start date and an end date.")
    start, end = map(pd.Timestamp, date_range)
    if start > end:
        raise ValueError("The start date must not follow the end date.")
    # A Boolean mask marks rows to keep. The county condition narrows that same mask.
    mask = df["Date"].between(start, end)
    if counties is not None:
        mask &= df["County"].isin(counties)
    # Return a separate table so later operations can safely work on the filtered rows.
    return df.loc[mask].copy().reset_index(drop=True)


def aggregate_monthly(df, method="Pooled population rate"):
    """Summarize the selected counties, using exact counts for pooled rates."""
    if method not in ["Pooled population rate", "Mean county rate"]:
        raise ValueError("Unknown aggregation method")
    # Sum counts and populations before calculating a pooled rate.
    grouped = df.groupby("Date", sort=True)
    result = grouped[["COVID_Cases", "Flu_Cases", "Population"]].sum()
    # Pooled rates weight counties by population. Mean county rates give them equal weight.
    for metric, count in [("COVID_Per_Capita", "COVID_Cases"), ("Flu_Per_Capita", "Flu_Cases")]:
        result[metric] = (result[count] / result["Population"] * 100000
                          if method == "Pooled population rate" else grouped[metric].mean())
    # The PM2.5 line gives each selected county equal weight in each month.
    result["PM25"] = grouped["PM25"].mean()
    return result.reset_index()


def make_scatter(df, x, y):
    """Use uniform markers; county color is optional to avoid 50-category legends."""
    if x not in X_VARIABLES or y not in Y_VARIABLES:
        raise ValueError("Unknown scatter variable")
    # Each point is one county-month. Color by county only for small selections.
    fig = px.scatter(df, x=x, y=y, color="County" if df.County.nunique() <= 10 else None,
                     hover_name="County", hover_data={"Date": "|%Y-%m-%d", "Population": ":,",
                     "COVID_Cases": True, "Flu_Cases": True}, labels=LABELS,
                     template="plotly_white", opacity=0.65)
    fig.update_traces(marker={"size": 7})
    fig.update_layout(title="Synthetic county-month observations", margin={"l": 20, "r": 20, "t": 55, "b": 20})
    return fig


def make_trends(summary, method):
    """Aligned panels share time, with separate disease and exposure axes."""
    # Share the month axis while giving case rates and PM2.5 separate scales.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=(f"Disease rates: {method.lower()}", "PM2.5: unweighted county mean"))
    for key, name, color in [("COVID_Per_Capita", "COVID-19", "#245b80"),
                             ("Flu_Per_Capita", "Influenza", "#b15c23")]:
        fig.add_trace(go.Scatter(x=summary.Date, y=summary[key], name=name,
                                line={"color": color}), row=1, col=1)
    fig.add_trace(go.Scatter(x=summary.Date, y=summary.PM25, name="PM2.5",
                            line={"color": "#386c48"}), row=2, col=1)
    fig.update_yaxes(title_text="Cases per 100,000 per month", row=1, col=1)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_layout(height=650, template="plotly_white", hovermode="x unified", legend={"orientation": "h"})
    return fig


def export_payload(df, state_name, date_range, selected_counties):
    """Serialize filtered values plus an auditable provenance manifest."""
    # Hash the exact download bytes and the running source for provenance.
    csv_bytes = df.to_csv(index=False, float_format="%.17g").encode("utf-8")
    # Record the inputs and environment so someone can trace how this download was produced.
    metadata = {"generator_version": GENERATOR_VERSION, "scenario_label": state_name,
                "scenario_settings": SCENARIOS[state_name],
                "dates": [str(x) for x in date_range], "counties": list(selected_counties),
                "rows": len(df), "columns": list(df.columns),
                "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "packages": {k: version(k) for k in ["numpy", "pandas", "plotly", "streamlit"]},
                "assumptions": {"count_distribution": "Poisson", "population_integer_bounds_inclusive": [20000, 500000],
                                "income_uniform_bounds": [35000, 100000], "county_multiplier_bounds": [0.8, 1.2],
                                "season_period_months": 12, "season_peak_month": "February",
                                "pm_season_amplitude": 0.15, "pm_noise_sd": 0.5, "pm_floor": 0.1,
                                "temperature_season_amplitude": 0.2, "temperature_noise_sd": 0.12,
                                "covid_season_amplitude": 0.25, "flu_season_amplitude": 0.65,
                                "environmental_disease_coupling": False},
                "disclaimer": "Entirely synthetic, uncalibrated illustrative scenarios. State labels are not state estimates."}
    return csv_bytes, json.dumps(metadata, indent=2).encode("utf-8")


# INTERFACE
# Streamlit reruns the script after widget changes. Cache the full dataset by scenario.
@st.cache_data
def load_scenario(state_name):
    return generate_synthetic_eco_health_data(state_name)


def main():
    # Configure the page before creating widgets or rendering any content.
    st.set_page_config(page_title="Eco-Health-Viz", page_icon="🌍", layout="wide", initial_sidebar_state="auto",
                       menu_items={"About": "Eco-Health-Viz 3.0. Synthetic scenario explorer by Mohammad Shaito, Ph.D."})
    st.title("🌍 Eco-Health-Viz")
    st.markdown("#### Synthetic environmental and respiratory disease scenarios")
    st.markdown("**Developed by [Mohammad Shaito, Ph.D.](https://sites.google.com/view/mohammad-shaito/home)**")
    st.info("All counties and observations are synthetic. State names label uncalibrated scenarios, not actual state conditions.")
    with st.expander("About Eco-Health-Viz"):
        st.markdown("""Explore 50 fictional counties per scenario over 36 months, January 2021 through December 2023.
Population and income remain constant within a county. Disease counts scale with population, with illustrative
seasonality and Poisson sampling. Environmental inputs do not enter the disease-rate equations.
These parameters are not estimated from observations.
Expected rates and counts are exported so the assumptions can be inspected. No epidemiological inference is performed.

Contact: [mohammad.shaito@mavs.uta.edu](mailto:mohammad.shaito@mavs.uta.edu)
or [mshaito78@gmail.com](mailto:mshaito78@gmail.com).""")
    with st.expander("Show Instructions"):
        st.markdown("""1. Choose a state-labeled synthetic scenario and an analysis mode.
2. Select both date endpoints. Only month-end observations within the interval are included.
3. Optionally select counties; an empty selection means all 50 fictional counties.
4. Environmental and disease series share seasonal patterns; visual associations are not evidence of an environmental effect.
5. Choose scatter variables or compare aligned monthly panels. Pooled rates use total cases divided by total
   selected population; mean county rates give each county equal weight.
6. Use inspection mode to download the full filtered CSV and its provenance JSON. The table shows the first 100 rows.""")
    # Use browser device signals so resizing a desktop window cannot show the note.
    headers = st.context.headers
    user_agent = headers.get("User-Agent", "")
    mobile_hint = headers.get("Sec-CH-UA-Mobile", "")
    mobile_device = mobile_hint == "?1" or re.search(
        r"Android|iPhone|iPad|iPod|Windows Phone|Mobile|Tablet|Silk",
        user_agent,
        re.IGNORECASE,
    )
    if mobile_device:
        st.info("📱 Tap the top-left sidebar icon to open the controls. Close the sidebar for more chart space.")

    # Apply the same scenario and filters before selecting a display mode.
    st.sidebar.header("Configuration and filtering")
    # Streamlit uses help= to put a small question-mark tooltip beside a control.
    state = st.sidebar.selectbox("State-labeled synthetic scenario", list(SCENARIOS), help="Choose one of five fictional scenarios. State names are labels for simulated data.")
    mode = st.sidebar.selectbox("Analysis mode", MODES, help="Compare variables in a scatter plot, view monthly trends, or inspect and download filtered records.")
    dates = st.sidebar.date_input("Date range", value=(pd.Timestamp("2021-01-01").date(), pd.Timestamp("2023-12-31").date()),
                                  min_value=pd.Timestamp("2021-01-01").date(), max_value=pd.Timestamp("2023-12-31").date(),
                                  help="Select both endpoints. The app includes month-end records within the selected dates.")
    full = load_scenario(state)
    # Key the county widget by scenario because each scenario has its own county names.
    counties = st.sidebar.multiselect("County subset (empty means all)", sorted(full.County.unique()), key=f"counties_{state}",
                                       help="Choose fictional counties to include. Leave this empty to use all 50 counties.")
    try:
        # Translate an empty widget selection to None, which means all counties.
        df = filter_records(full, dates, counties or None)
    except ValueError as exc:
        st.info(str(exc))
        st.stop()
    if df.empty:
        st.warning("No month-end observations in the selected interval.")
        st.stop()
    st.caption(f"{state} scenario | {len(df):,} simulated records | {df.County.nunique()} fictional counties | generator {GENERATOR_VERSION}")
    # Build only the chart or table selected in the analysis menu.
    if mode == MODES[0]:
        x = st.selectbox("Environmental or economic variable", X_VARIABLES, format_func=LABELS.get, help="Choose PM2.5, temperature offset, or household income for the horizontal axis.")
        y = st.selectbox("Disease metric", Y_VARIABLES, format_func=LABELS.get, help="Choose simulated COVID-19 or influenza rates per 100,000 people per month for the vertical axis.")
        st.plotly_chart(make_scatter(df, x, y), width="stretch")
        st.caption("Uniform marker sizes. County colors appear only when 10 or fewer counties are selected. Hover includes date and population.")
    elif mode == MODES[1]:
        method = st.selectbox("Disease-rate aggregation", ["Pooled population rate", "Mean county rate"],
                              help="Pooled rates divide total cases by total population. Mean county rates give each county equal weight.")
        st.plotly_chart(make_trends(aggregate_monthly(df, method), method), width="stretch")
        st.caption("Summaries refer to the selected fictional counties, not a real state. Separate panels avoid overlaying unrelated axes.")
    else:
        st.subheader("Filtered synthetic records")
        # Preview 100 rows for readability. Build downloads from the full filtered table.
        st.dataframe(df.head(100), width="stretch")
        csv_bytes, meta = export_payload(df, state, dates, sorted(df.County.unique()))
        st.download_button("Download filtered CSV", csv_bytes, f"eco_health_{state.lower()}_v3.csv", "text/csv", help="Download every record that matches the current filters.")
        st.download_button("Download provenance JSON", meta, f"eco_health_{state.lower()}_v3_metadata.json", "application/json", help="Download scenario settings, filter details, software versions, and checksums.")
    st.caption("Eco-Health-Viz | Synthetic illustration, not observed surveillance or validated epidemiology")


# Run the interface as a script. Imports can use the functions directly.
if __name__ == "__main__":
    main()
