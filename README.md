# Eco-Health-Viz 🌍

An interactive web-based dashboard for spatiotemporal analysis of environmental stressors (temperature anomalies, $\text{PM}_{2.5}$) and infectious disease dynamics (COVID-19, Influenza) across U.S. counties.

## Features
- **Multi-Pathogen & Environmental Integration:** Correlate air quality indices and climate anomalies directly with epidemiological trends.
- **Dynamic Streamlit Frontend:** Filter by state, date range, and variable pairs with responsive Plotly visualizations.
- **Synthetic Data Engine:** Graceful fallback simulation when live sensor feeds or public health records are incomplete.

## Quickstart

```bash
# Clone the repository
git clone [https://github.com/YOUR-USERNAME/Eco-Health-Viz.git](https://github.com/YOUR-USERNAME/Eco-Health-Viz.git)
cd Eco-Health-Viz

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
