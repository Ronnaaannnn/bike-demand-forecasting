import os
import random
from datetime import date

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

import model as M

app = Flask(__name__)
app.secret_key = "change-me"
CSV = os.path.join(M.BASE, "hour.csv")

clf, metrics = M.load()


def ensure_model():
    """Train automatically on first start if hour.csv is present."""
    global clf, metrics
    if clf is None and os.path.exists(CSV):
        M.train(CSV)
        clf, metrics = M.load()


ensure_model()


def season_of(d):
    """Capital Bikeshare seasons: 1 spring, 2 summer, 3 fall, 4 winter."""
    md = (d.month, d.day)
    if md >= (12, 21) or md < (3, 20):
        return 4
    if md < (6, 21):
        return 1
    if md < (9, 23):
        return 2
    return 3


def level(v):
    if v < metrics["low_cut"]:
        return "Low"
    return "Medium" if v < metrics["high_cut"] else "High"


def build_rows(form, hours):
    """Turn user-friendly input into the model's 12 features (one row per hour)."""
    try:
        d = date.fromisoformat(form.get("date", ""))
    except ValueError:
        raise ValueError("Please choose a valid date.")
    try:
        temp = float(form["temp"]); atemp = float(form["atemp"])
        hum = float(form["hum"]); wind = float(form["wind"])
        weather = int(form["weathersit"])
        hr = int(form["hr"])
    except (KeyError, ValueError):
        raise ValueError("All fields must be filled with numbers.")
    if not -10 <= temp <= 41 or not -16 <= atemp <= 50:
        raise ValueError("Temperature must be between -10 and 41 °C (feels-like up to 50 °C).")
    if not 0 <= hum <= 100:
        raise ValueError("Humidity must be 0–100 %.")
    if not 0 <= wind <= 67:
        raise ValueError("Wind speed must be 0–67 km/h.")
    if weather not in (1, 2, 3, 4):
        raise ValueError("Pick a weather type.")
    if not 0 <= hr <= 23:
        raise ValueError("Hour must be 0–23.")

    holiday = 1 if form.get("holiday") else 0
    weekday = (d.weekday() + 1) % 7            # dataset: 0 = Sunday
    working = 1 if (weekday not in (0, 6) and not holiday) else 0
    base = {"season": season_of(d), "yr": 1 if d.year >= 2012 else 0, "mnth": d.month,
            "holiday": holiday, "weekday": weekday, "workingday": working,
            "weathersit": weather, "temp": temp / M.TEMP_MAX, "atemp": atemp / M.ATEMP_MAX,
            "hum": hum / M.HUM_MAX, "windspeed": wind / M.WIND_MAX}
    rows = [dict(base, hr=h) for h in (hours if hours is not None else [hr])]
    return d, hr, pd.DataFrame(rows)[M.FEATURES]


@app.route("/")
def index():
    return render_template("index.html", m=metrics, csv_ok=os.path.exists(CSV))


@app.route("/train", methods=["POST"])
def train():
    global clf, metrics
    if not os.path.exists(CSV):
        flash("hour.csv not found next to app.py.", "err")
        return redirect(url_for("index"))
    M.train(CSV)
    clf, metrics = M.load()
    flash("Models retrained and the best one was selected.", "ok")
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, values, error = None, {}, None
    if request.method == "POST":
        values = request.form.to_dict()
        if clf is None:
            error = "No trained model yet. Train it on the dashboard first."
        else:
            try:
                d, hr, X = build_rows(request.form, hours=list(range(24)))
                day_pred = [max(0, round(float(p))) for p in clf.predict(X)]
                result = {"value": day_pred[hr], "level": level(day_pred[hr]),
                          "hr": hr, "date": d.strftime("%A, %d %B %Y"),
                          "day": day_pred, "day_total": sum(day_pred),
                          "peak_hr": day_pred.index(max(day_pred)),
                          "max": max(day_pred) or 1}
            except ValueError as e:
                error = str(e)
    return render_template("predict.html", values=values, result=result, error=error,
                           trained=clf is not None, today=date.today().isoformat())


@app.route("/sample")
def sample():
    """Return a random real hour from the 30% test set, in friendly units."""
    if not os.path.exists(M.SAMPLES_PATH):
        return jsonify({"error": "train first"}), 404
    r = pd.read_csv(M.SAMPLES_PATH).sample(1).iloc[0]
    return jsonify({"date": r["dteday"], "hr": int(r["hr"]), "weathersit": int(r["weathersit"]),
                    "temp": round(r["temp"] * M.TEMP_MAX, 1), "atemp": round(r["atemp"] * M.ATEMP_MAX, 1),
                    "hum": round(r["hum"] * M.HUM_MAX), "wind": round(r["windspeed"] * M.WIND_MAX, 1),
                    "holiday": int(r["holiday"]), "actual": int(r["cnt"])})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
