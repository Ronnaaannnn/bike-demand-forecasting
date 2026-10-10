#  Bike Sharing Demand Forecasting: Flask + Docker

An end-to-end machine learning project that forecasts **hourly bike-rental demand** and serves the model through a Flask web app, with a training dashboard, evaluation graphs and a live forecasting form. Fully containerized with Docker.

**Dataset:** [Bike Sharing Dataset (UCI / Kaggle)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Capital Bikeshare, Washington D.C., 2011–2012 (Fanaee-T & Gama, 2013).

## Problem
Bike-sharing operators need to know **how many bikes will be rented at a given hour** so they can rebalance stations, schedule maintenance and avoid empty or overflowing docks. Demand depends heavily on the hour, day type, season and weather. The dataset has **17,379 hourly records** (731 days), with a mean of ~190 rentals/hour and a peak of 977.

Features used: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`.
Target: `cnt` (total rentals in that hour).

> `casual` and `registered` are **deliberately excluded**: they sum to `cnt`, so using them would be target leakage.

##  Approach
1. Load `hour.csv` and sort chronologically
2. **70/30 chronological split**: the first 70% of time trains the model, the last 30% (≈ Aug–Dec 2012) tests it. This mimics real forecasting (predict the future from the past) instead of a random shuffle that leaks neighbouring hours
3. Train and compare **Linear Regression, Random Forest and Gradient Boosting**
4. Automatically pick the model with the lowest RMSE on the test set
5. Save the model and metrics, then serve everything through Flask

## 📈 Results (30% future test set)
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 146.43 | 184.59 | 0.298 |
| Random Forest | 48.83 | 74.30 | 0.886 |
| **Gradient Boosting** ✅ | 49.49 | **73.76** | **0.888** |

The linear baseline struggles (R² 0.30) because demand is a non-linear function of the hour (twin commute peaks at 8:00 and 17:00), so tree ensembles win by a wide margin. The best model is off by about **49 bikes per hour** on average, on a mean demand of ~190.

## Graphs generated
- Demand by hour (working day vs weekend)
- Weekly demand 2011–2012 with the train/test cut
- Forecast vs actual (last 14 days of the test set)
- Actual vs predicted scatter
- Residual distribution
- Model comparison (RMSE)
- Feature importance
- Temperature & weather vs demand
- Weekday × hour heatmap

## Web app features
| Page | What it does |
|------|--------------|
| **Dashboard (`/`)** | Metric cards, chronological split info, model comparison table, all graphs, **Train / Retrain** button |
| **Forecast demand (`/predict`)** | Pick a date, hour and weather, and get the forecast for that hour (Low / Medium / High demand), plus a **24-hour demand curve**, the day's total and the peak hour. A button loads a real example from the test set so you can compare with the true value |

The form takes friendly units (°C, %, km/h, a date picker). Season, weekday, working day and the dataset's normalization are handled automatically, and input is validated.

##  Project structure
```
bike_demand/
├── app.py                # Flask routes (/, /train, /predict, /sample)
├── model.py              # Training, model comparison, graph generation
├── hour.csv, day.csv     # dataset
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── templates/            # base.html, index.html, predict.html
├── static/plots/         # generated graphs
└── artifacts/            # model.joblib, metrics.json, test_samples.csv
```

##  Run with Docker (recommended)
```bash
docker compose up --build
```
Open **http://localhost:5000**. On first start the models train automatically (about a minute), then the dashboard loads.

Or without Compose:
```bash
docker build -t bike-demand .
docker run -p 5000:5000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/static/plots:/app/static/plots \
  bike-demand
```

## Run without Docker
```bash
pip install -r requirements.txt
python model.py     # train first
python app.py       # open http://127.0.0.1:5000
```

## Future improvements
- Add lag features (demand 1 h / 24 h / 168 h ago) and rolling averages
- Try XGBoost / LightGBM and time-series cross-validation
- Real weather forecast API so predictions need no manual weather input
- Predict `casual` and `registered` users separately
- Deploy the container to Render / AWS ECS / Google Cloud Run

##  Tech stack
Python · Flask · scikit-learn · pandas · NumPy · Matplotlib · joblib · Jinja2 · Docker · Gunicorn

##  License & credits
Dataset: Hadi Fanaee-T, LIAAD, University of Porto. Cite: Fanaee-T & Gama, "Event labeling combining ensemble detectors and background knowledge", *Progress in Artificial Intelligence* (2013).
