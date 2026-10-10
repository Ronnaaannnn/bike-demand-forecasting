"""Training pipeline: chronological 70/30 split, 3 regressors compared, metrics + graphs."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")
PLOTS = os.path.join(BASE, "static", "plots")
MODEL_PATH = os.path.join(ART, "model.joblib")
METRICS_PATH = os.path.join(ART, "metrics.json")
SAMPLES_PATH = os.path.join(ART, "test_samples.csv")

# casual / registered are NOT used: they add up to cnt (target leakage)
FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"
# data is normalised: temp/41, atemp/50, hum/100, windspeed/67
TEMP_MAX, ATEMP_MAX, HUM_MAX, WIND_MAX = 41, 50, 100, 67

BG, FG, ACC, ACC2, GRN = "#0f172a", "#e2e8f0", "#38bdf8", "#f59e0b", "#22c55e"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, pad=10)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color("#334155")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def _legend(ax):
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#334155")


def _save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=110, facecolor=BG)
    plt.close(fig)


def _scores(y, p):
    p = np.clip(p, 0, None)
    return {"mae": round(float(mean_absolute_error(y, p)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "r2": round(float(r2_score(y, p)), 4)}


def train(csv_path):
    os.makedirs(ART, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values(["dteday", "hr"]).reset_index(drop=True)
    df["dteday"] = pd.to_datetime(df["dteday"])

    # 70 / 30 CHRONOLOGICAL split: train on the past, forecast the future
    cut = int(len(df) * 0.70)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    X_tr, y_tr, X_te, y_te = tr[FEATURES], tr[TARGET], te[FEATURES], te[TARGET]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42),
    }
    results, preds = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds[name] = np.clip(m.predict(X_te), 0, None)
        results[name] = _scores(y_te, preds[name])

    best = min(results, key=lambda k: results[k]["rmse"])
    clf, pred = models[best], preds[best]
    s = results[best]
    mape = float(np.mean(np.abs(y_te - pred) / y_te) * 100)  # cnt >= 1 so safe

    q1, q2 = np.percentile(y_tr, [33, 66])
    metrics = {
        "rows": int(len(df)), "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "train_from": str(tr.dteday.min().date()), "train_to": str(tr.dteday.max().date()),
        "test_from": str(te.dteday.min().date()), "test_to": str(te.dteday.max().date()),
        "best_model": best, "mae": s["mae"], "rmse": s["rmse"], "r2": s["r2"],
        "mape": round(mape, 1), "avg_demand": round(float(df[TARGET].mean()), 1),
        "peak_demand": int(df[TARGET].max()),
        "comparison": results, "low_cut": float(q1), "high_cut": float(q2),
    }

    # ---------- graphs ----------
    # 1. average demand by hour: working vs non-working days
    fig, ax = plt.subplots(figsize=(5, 4))
    for wd, c, lab in [(1, ACC, "Working day"), (0, ACC2, "Weekend / holiday")]:
        h = df[df.workingday == wd].groupby("hr")[TARGET].mean()
        ax.plot(h.index, h.values, color=c, lw=2.2, label=lab)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Avg rentals / hour")
    _legend(ax); _style(ax, "Demand by hour of day")
    _save(fig, "by_hour.png")

    # 2. total demand over time (weekly) + train/test cut
    wk = df.set_index("dteday")[TARGET].resample("W").sum()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(wk.index, wk.values, color=ACC, lw=1.8)
    ax.axvline(te.dteday.min(), color=ACC2, ls="--", label="Train | Test cut")
    ax.set_ylabel("Rentals / week"); _legend(ax)
    plt.setp(ax.get_xticklabels(), rotation=30)
    _style(ax, "Weekly demand 2011-2012")
    _save(fig, "weekly.png")

    # 3. actual vs predicted: last 14 days of the test set
    last = te.tail(24 * 14)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(len(last)), last[TARGET].values, color=FG, lw=1.4, label="Actual")
    ax.plot(range(len(last)), pred[-len(last):], color=ACC2, lw=1.4, label=f"Predicted")
    ax.set_xlabel("Hours (last 14 days of test set)"); ax.set_ylabel("Rentals")
    _legend(ax); _style(ax, f"Forecast vs actual · {best}")
    _save(fig, "forecast.png")

    # 4. actual vs predicted scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_te, pred, s=5, alpha=.35, color=ACC)
    mx = max(y_te.max(), pred.max())
    ax.plot([0, mx], [0, mx], color=ACC2, ls="--", label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); _legend(ax)
    _style(ax, f"Actual vs predicted (R² = {s['r2']})")
    _save(fig, "scatter.png")

    # 5. residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_te.values - pred, bins=50, color=ACC)
    ax.axvline(0, color=ACC2, ls="--")
    ax.set_xlabel("Actual − predicted"); ax.set_ylabel("Hours")
    _style(ax, "Residual distribution")
    _save(fig, "residuals.png")

    # 6. model comparison (RMSE)
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(results)
    vals = [results[n]["rmse"] for n in names]
    bars = ax.bar(range(3), vals, color=[GRN if n == best else "#475569" for n in names])
    ax.set_xticks(range(3), [n.replace(" ", "\n") for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", color=FG)
    ax.set_ylabel("RMSE (lower is better)")
    _style(ax, "Model comparison")
    _save(fig, "comparison.png")

    # 7. feature importance (tree models only)
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4.4))
        ax.barh(imp.index, imp.values, color=ACC)
        _style(ax, "Feature importance")
        _save(fig, "importance.png")

    # 8. demand vs temperature, coloured by weather
    fig, ax = plt.subplots(figsize=(5, 4))
    for w, c, lab in [(1, ACC, "Clear"), (2, ACC2, "Mist/cloudy"), (3, "#f43f5e", "Light rain/snow")]:
        d = df[df.weathersit == w]
        ax.scatter(d.temp * TEMP_MAX, d[TARGET], s=4, alpha=.3, color=c, label=lab)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Rentals / hour")
    _legend(ax); _style(ax, "Temperature & weather vs demand")
    _save(fig, "temp_weather.png")

    # 9. weekday x hour heatmap
    piv = df.pivot_table(index="weekday", columns="hr", values=TARGET, aggfunc="mean")
    piv.index = [["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][i] for i in piv.index]
    piv = piv.loc[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_yticks(range(7), piv.index); ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("Hour of day")
    _style(ax, "Avg demand: weekday × hour")
    _save(fig, "heatmap.png")

    # ---------- persist ----------
    joblib.dump(clf, MODEL_PATH)
    te.sample(600, random_state=1)[["dteday"] + FEATURES + [TARGET]].to_csv(
        SAMPLES_PATH, index=False)  # used by the "load example" button
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    return metrics


def load():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        return None, None
    with open(METRICS_PATH) as f:
        return joblib.load(MODEL_PATH), json.load(f)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "hour.csv")
    m = train(path)
    print(f"Best: {m['best_model']}  MAE={m['mae']}  RMSE={m['rmse']}  "
          f"R2={m['r2']}  MAPE={m['mape']}%")
    for k, v in m["comparison"].items():
        print(f"  {k:18s} {v}")
