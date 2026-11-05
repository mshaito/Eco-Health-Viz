# Eco-Health-Viz

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io/)

## Overview

Eco-Health-Viz is an interactive dashboard for exploring simulated air quality, temperature, household income, and respiratory disease rates. Each of the five state-labeled scenarios contains 50 fictional counties with monthly records from January 2021 through December 2023. The state labels identify simulation settings. Shared seasonality can produce visible associations between environmental and disease series.

## Live demo

[Open Eco-Health-Viz](https://eco-health-viz.streamlit.app/)

## Documentation

The [Eco-Health-Viz User Manual](Eco_Health_Viz_User_Manual.pdf) covers the sidebar filters, chart controls, rate calculations, downloads, and common troubleshooting steps.

## Explore the dashboard

1. In the sidebar, choose a **State-labeled synthetic scenario**, an **Analysis mode**, and both endpoints of the **Date range**. Leave **County subset** empty to include all counties.
2. Choose **Environmental and disease comparison** to plot PM2.5, temperature offset, or household income against simulated COVID-19 or influenza rates.
3. Choose **Monthly trends** to view disease rates and mean PM2.5 on a shared month axis. Select **Pooled population rate** or **Mean county rate** to change how disease rates are summarized.
4. Choose **Data inspection and export** to preview up to 100 filtered rows and download the complete filtered CSV and provenance JSON.

The **?** beside a control explains its purpose. On a phone, use the top-left sidebar icon to reach the filters.

## Run locally

From the directory containing `app.py` and `requirements.txt`:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Reproducibility

Each scenario uses fixed random seeds. The provenance JSON records the scenario, date range, counties, software versions, and checksums. Keep it with the matching CSV. The supplementary verification package accompanies the manuscript and includes the checks for the evaluated local release.

## Contact

Developed by Mohammad Shaito, Ph.D. [Email](mailto:mohammad.shaito@mavs.uta.edu)
