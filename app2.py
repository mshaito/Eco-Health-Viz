"""Eco-Health-Viz 2.0: transparent synthetic scenarios, not observed surveillance.

The pure functions above main() are independently importable and testable.
Run with: streamlit run 'app(2).py'
"""
from pathlib import Path
import hashlib
import io
import json
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

GENERATOR_VERSION = "2.0.0"
# All values below are arbitrary scenario settings, not state estimates.
SCENARIOS = {
    "Texas": {"seed": 4101, "pm": 8.0, "temp": 1.0, "covid": 100.0, "flu": 60.0},
    "California": {"seed": 4102, "pm": 9.0, "temp": 1.2, "covid": 120.0, "flu": 55.0},
    "New York": {"seed": 4103, "pm": 7.0, "temp": 0.8, "covid": 90.0, "flu": 65.0},
    "Florida": {"seed": 4104, "pm": 6.0, "temp": 1.4, "covid": 110.0, "flu": 50.0},
    "Illinois": {"seed": 4105, "pm": 8.5, "temp": 0.9, "covid": 105.0, "flu": 70.0},
}
MODES = ["Environmental and disease comparison", "Monthly trends", "Data inspection and export"]
X_VARIABLES = ["PM25", "Temperature_Anomaly", "Median_Household_Income"]
Y_VARIABLES = ["COVID_Per_Capita", "Flu_Per_Capita"]
LABELS = {"PM25": "Simulated PM2.5 (µg/m³)",
          "Temperature_Anomaly": "Simulated temperature offset (°C)",
          "Median_Household_Income": "Synthetic household income (nominal units)",
          "COVID_Per_Capita": "Simulated COVID-19 cases per 100,000 per month",
          "Flu_Per_Capita": "Simulated influenza cases per 100,000 per month"}


def generate_synthetic_eco_health_data(state_name, environmental_effects=True):
    """Return a deterministic scenario and its known generating quantities.

    Stable county population/income and risk are sampled once. Environmental
    and count streams are separate, so toggling assumed effects preserves all
    environmental and demographic inputs. No coefficients are empirically fitted.
    """
    cfg = SCENARIOS[state_name]
    env_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0]))
    count_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1]))
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="ME")
    rows = []
    for i in range(50):
        population = int(np.clip(env_rng.lognormal(np.log(120000), 0.9), 20000, 1500000))
        income = round(float(env_rng.lognormal(np.log(65000), 0.25)), 2)
        risk = float(env_rng.lognormal(0.0, 0.2))
        pm_base = float(cfg["pm"] * env_rng.lognormal(0.0, 0.25))
        pm_error = temp_error = 0.0
        for t, date in enumerate(dates):
            season = float(np.cos(2 * np.pi * (t - 1) / 12))
            pm_error = 0.7 * pm_error + float(env_rng.normal(0, 0.5))
            temp_error = 0.7 * temp_error + float(env_rng.normal(0, 0.12))
            pm = max(0.1, pm_base * (1 + 0.15 * season) + pm_error)
            temp = cfg["temp"] + 0.2 * season + temp_error
            coupling = 1.0 if environmental_effects else 0.0
            covid_rate = cfg["covid"] * risk * (1 + 0.25 * season) * np.exp(
                coupling * (0.12 * (pm - cfg["pm"]) / 10 + 0.04 * (temp - cfg["temp"])))
            flu_rate = cfg["flu"] * risk * (1 + 0.65 * season) * np.exp(
                coupling * (0.08 * (pm - cfg["pm"]) / 10 - 0.03 * (temp - cfg["temp"])))
            covid_mu = population * covid_rate / 100000
            flu_mu = population * flu_rate / 100000
            # NB shape k=20: mean mu, variance mu + mu²/k.
            covid = int(count_rng.negative_binomial(20, 20 / (20 + covid_mu)))
            flu = int(count_rng.negative_binomial(20, 20 / (20 + flu_mu)))
            rows.append({"Date": date, "State": state_name,
                         "County": f"{state_name} Synthetic County {i + 1:02d}",
                         "Population": population, "Median_Household_Income": income,
                         "Temperature_Anomaly": temp, "PM25": pm,
                         "COVID_Cases": covid, "Flu_Cases": flu,
                         "COVID_Per_Capita": covid / population * 100000,
                         "Flu_Per_Capita": flu / population * 100000,
                         "Expected_COVID_Rate": float(covid_rate), "Expected_Flu_Rate": float(flu_rate),
                         "Expected_COVID_Count": float(covid_mu), "Expected_Flu_Count": float(flu_mu),
                         "County_Risk": risk, "County_PM_Baseline": pm_base,
                         "Seasonal_Cosine": season, "Generator_Version": GENERATOR_VERSION,
                         "Scenario_Seed": cfg["seed"], "Environmental_Effects": bool(environmental_effects)})
    return pd.DataFrame(rows).sort_values(["Date", "County"]).reset_index(drop=True)


def filter_records(df, date_range, counties=None):
    """Apply inclusive dates and an optional county subset; reject partial ranges."""
    if len(date_range) != 2:
        raise ValueError("Select both a start date and an end date.")
    start, end = map(pd.Timestamp, date_range)
    if start > end:
        raise ValueError("The start date must not follow the end date.")
    mask = df["Date"].between(start, end)
    if counties is not None:
        mask &= df["County"].isin(counties)
    return df.loc[mask].copy().reset_index(drop=True)


def aggregate_monthly(df, method="Pooled population rate"):
    """Summarize the selected counties, using exact counts for pooled rates."""
    if method not in ["Pooled population rate", "Mean county rate"]:
        raise ValueError("Unknown aggregation method")
    grouped = df.groupby("Date", sort=True)
    result = grouped[["COVID_Cases", "Flu_Cases", "Population"]].sum()
    for metric, count in [("COVID_Per_Capita", "COVID_Cases"), ("Flu_Per_Capita", "Flu_Cases")]:
        result[metric] = (result[count] / result["Population"] * 100000
                          if method == "Pooled population rate" else grouped[metric].mean())
    result["PM25"] = grouped["PM25"].mean()
    return result.reset_index()


def make_scatter(df, x, y):
    """Use uniform markers; county color is optional to avoid 50-category legends."""
    if x not in X_VARIABLES or y not in Y_VARIABLES:
        raise ValueError("Unknown scatter variable")
    fig = px.scatter(df, x=x, y=y, color="County" if df.County.nunique() <= 10 else None,
                     hover_name="County", hover_data={"Date": "|%Y-%m-%d", "Population": ":,",
                     "COVID_Cases": True, "Flu_Cases": True}, labels=LABELS,
                     template="plotly_white", opacity=0.65)
    fig.update_traces(marker={"size": 7})
    fig.update_layout(title="Synthetic county-month observations", margin={"l": 20, "r": 20, "t": 55, "b": 20})
    return fig


def make_trends(summary, method):
    """Aligned panels share time, with separate disease and exposure axes."""
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


def export_payload(df, state_name, environmental_effects, date_range, selected_counties):
    """Serialize filtered values plus an auditable provenance manifest."""
    csv_bytes = df.to_csv(index=False, float_format="%.17g").encode("utf-8")
    metadata = {"generator_version": GENERATOR_VERSION, "scenario_label": state_name,
                "scenario_settings": SCENARIOS[state_name], "environmental_effects": bool(environmental_effects),
                "dates": [str(x) for x in date_range], "counties": list(selected_counties),
                "rows": len(df), "columns": list(df.columns),
                "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "packages": {k: version(k) for k in ["numpy", "pandas", "plotly", "streamlit"]},
                "assumptions": {"negative_binomial_shape": 20, "ar_coefficient": 0.7,
                                "covid_pm_beta_per_10_units": 0.12, "covid_temp_beta": 0.04,
                                "flu_pm_beta_per_10_units": 0.08, "flu_temp_beta": -0.03},
                "disclaimer": "Entirely synthetic, uncalibrated illustrative scenarios. State labels are not state estimates."}
    return csv_bytes, json.dumps(metadata, indent=2).encode("utf-8")


# INTERFACE
@st.cache_data
def load_scenario(state_name, environmental_effects):
    return generate_synthetic_eco_health_data(state_name, environmental_effects)


def main():
    st.set_page_config(page_title="Eco-Health-Viz", page_icon="🌍", layout="wide", initial_sidebar_state="auto",
                       menu_items={"About": "Eco-Health-Viz 2.0. Synthetic scenario explorer by Mohammad Shaito, Ph.D."})
    st.title("🌍 Eco-Health-Viz")
    st.markdown("#### Synthetic environmental and respiratory disease scenarios")
    st.markdown("**Developed by [Mohammad Shaito, Ph.D.](https://sites.google.com/view/mohammad-shaito/home)**")
    st.info("All counties and observations are synthetic. State names label uncalibrated scenarios, not actual state conditions.")
    with st.expander("About Eco-Health-Viz"):
        st.markdown("""Explore 50 fictional counties per scenario over 36 months, January 2021 through December 2023.
Population and income remain constant within a county. Disease counts scale with population, with illustrative
seasonality and optional assumed environmental effects. These parameters are not estimated from observations.
Expected rates and counts are exported so the assumptions can be inspected. No epidemiological inference is performed.

Contact: [mohammad.shaito@mavs.uta.edu](mailto:mohammad.shaito@mavs.uta.edu)
or [mshaito78@gmail.com](mailto:mshaito78@gmail.com).""")
    with st.expander("Show Instructions"):
        st.markdown("""1. Choose a state-labeled synthetic scenario and an analysis mode.
2. Select both date endpoints. Only month-end observations within the interval are included.
3. Optionally select counties; an empty selection means all 50 fictional counties.
4. Toggle the assumed environmental effects. Turning them off preserves the environmental inputs;
   seasonal patterns remain and correlations can still occur.
5. Choose scatter variables or compare aligned monthly panels. Pooled rates use total cases divided by total
   selected population; mean county rates give each county equal weight.
6. Use inspection mode to download the full filtered CSV and its provenance JSON. The table shows the first 100 rows.""")
    with st.expander("📱 Access on a Mobile Device"):
        st.write("Open the deployed address in a phone or tablet browser. Tap the top-left sidebar icon to open controls. Close it for more chart space; landscape orientation may help.")
    st.markdown('<style>.mobile-tip{display:none}@media(max-width:768px){.mobile-tip{display:block;padding:8px 0}}</style><div class="mobile-tip">📱 Open the top-left sidebar icon for controls.</div>', unsafe_allow_html=True)
    st.sidebar.header("Configuration and filtering")
    state = st.sidebar.selectbox("State-labeled synthetic scenario", list(SCENARIOS))
    mode = st.sidebar.selectbox("Analysis mode", MODES)
    dates = st.sidebar.date_input("Date range", value=(pd.Timestamp("2021-01-01").date(), pd.Timestamp("2023-12-31").date()),
                                  min_value=pd.Timestamp("2021-01-01").date(), max_value=pd.Timestamp("2023-12-31").date())
    effects = st.sidebar.checkbox("Include assumed environmental effects", value=True)
    full = load_scenario(state, effects)
    counties = st.sidebar.multiselect("County subset (empty means all)", sorted(full.County.unique()), key=f"counties_{state}")
    try:
        df = filter_records(full, dates, counties or None)
    except ValueError as exc:
        st.info(str(exc)); st.stop()
    if df.empty:
        st.warning("No month-end observations in the selected interval."); st.stop()
    st.caption(f"{state} scenario | {len(df):,} simulated records | {df.County.nunique()} fictional counties | generator {GENERATOR_VERSION}")
    if mode == MODES[0]:
        x = st.selectbox("Environmental or economic variable", X_VARIABLES, format_func=LABELS.get)
        y = st.selectbox("Disease metric", Y_VARIABLES, format_func=LABELS.get)
        st.plotly_chart(make_scatter(df, x, y), width="stretch")
        st.caption("Uniform marker sizes. County colors appear only when 10 or fewer counties are selected. Hover includes date and population.")
    elif mode == MODES[1]:
        method = st.selectbox("Disease-rate aggregation", ["Pooled population rate", "Mean county rate"])
        st.plotly_chart(make_trends(aggregate_monthly(df, method), method), width="stretch")
        st.caption("Summaries refer to the selected fictional counties, not a real state. Separate panels avoid overlaying unrelated axes.")
    else:
        st.subheader("Filtered synthetic records")
        st.dataframe(df.head(100), width="stretch")
        csv_bytes, meta = export_payload(df, state, effects, dates, sorted(df.County.unique()))
        st.download_button("Download filtered CSV", csv_bytes, f"eco_health_{state.lower()}_v2.csv", "text/csv")
        st.download_button("Download provenance JSON", meta, f"eco_health_{state.lower()}_v2_metadata.json", "application/json")
    st.caption("Eco-Health-Viz | Synthetic illustration, not observed surveillance or validated epidemiology")


if __name__ == "__main__":
    main()
