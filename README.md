# traffic_viz

Interactive probabilistic travel-time analysis for the I-5 Northbound corridor
using PeMS 5-minute station data.

## Features

- Corridor-level travel time estimation
- Time-of-day and weekday/weekend conditioning
- Probabilistic metrics (mean, p95, BTI, reliability)
- Interactive Streamlit dashboard
- CLI for preprocessing and diagnostics

https://drive.google.com/drive/folders/1Ms151kBdxH-d284sY8oyMqVU61OwpqXG?usp=sharing

## Quick start

```bash
# install
pip install -e .

# preprocess raw PeMS data (first time only)
traffic_viz preprocess --data-dir /path/to/data

# run the app
traffic_viz app --data-dir /path/to/data
```
