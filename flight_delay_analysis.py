#!/usr/bin/env python3
"""
Flight Delay Analysis (BTS aggregated data)

This script loads the BTS monthly aggregated delay dataset, performs cleaning and
feature engineering, runs EDA and saves plots, trains a simple model to predict
`delay_rate`, and writes artifacts (cleaned data, metrics, model, importances).

It is designed to be robust in minimal environments:
- Ensures output directories exist
- Falls back to CSV if Parquet writer is unavailable
- Uses a non-interactive matplotlib backend
- Prints explicit paths for all saved artifacts

Expected columns in input CSV:
  ['year','month','carrier','carrier_name','airport','airport_name',
   'arr_flights','arr_del15','carrier_ct','weather_ct','nas_ct','security_ct','late_aircraft_ct',
   'arr_cancelled','arr_diverted','arr_delay','carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# ----------------------------
# Utilities
# ----------------------------

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    wsum = np.nansum(weights.values)
    if wsum <= 0 or np.isnan(wsum):
        return float("nan")
    return float(np.nansum(values.values * weights.values) / wsum)


def month_sin_cos(month: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    radians = 2 * math.pi * (month.astype(int) - 1) / 12.0
    return np.sin(radians), np.cos(radians)


def year_month_int(year: pd.Series, month: pd.Series) -> np.ndarray:
    return (year.astype(int) * 100 + month.astype(int)).values


def save_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Try saving as Parquet; fallback to CSV if pyarrow/fastparquet is missing.
    Returns the actual path written.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as e:
        # Fallback to CSV next to parquet path
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


# ----------------------------
# Data loading and cleaning
# ----------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = {
        "year", "month", "carrier", "carrier_name", "airport", "airport_name",
        "arr_flights", "arr_del15", "arr_cancelled", "arr_diverted", "arr_delay",
        "carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct",
        "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
    }
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    count_cols = [
        "arr_flights", "arr_del15", "arr_cancelled", "arr_diverted",
        "carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct",
    ]
    delay_cols = [
        "arr_delay", "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
    ]
    df[count_cols] = df[count_cols].fillna(0.0)
    df[delay_cols] = df[delay_cols].fillna(0.0)

    df = df[df["arr_flights"] > 0].copy()

    df["delay_rate"] = np.where(df["arr_flights"] > 0, df["arr_del15"] / df["arr_flights"], np.nan)
    df["cancel_rate"] = np.where(df["arr_flights"] > 0, df["arr_cancelled"] / df["arr_flights"], np.nan)
    df["divert_rate"] = np.where(df["arr_flights"] > 0, df["arr_diverted"] / df["arr_flights"], np.nan)
    df["avg_arr_delay_per_flight"] = np.where(df["arr_flights"] > 0, df["arr_delay"] / df["arr_flights"], np.nan)
    df["avg_delay_per_delayed_flight"] = np.where(df["arr_del15"] > 0, df["arr_delay"] / df["arr_del15"], np.nan)

    for c in ["carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct"]:
        df[f"{c}_share"] = np.where(df["arr_del15"] > 0, df[c] / df["arr_del15"], np.nan)

    for c in ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]:
        df[f"{c}_share_minutes"] = np.where(df["arr_delay"] > 0, df[c] / df["arr_delay"], np.nan)

    df["ym"] = year_month_int(df["year"], df["month"]) 

    sin_m, cos_m = month_sin_cos(df["month"])
    df["month_sin"] = sin_m
    df["month_cos"] = cos_m
    df["arr_flights_log1p"] = np.log1p(df["arr_flights"]) 

    return df


# ----------------------------
# EDA plots
# ----------------------------

def plot_delay_distribution(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["delay_rate"].dropna(), bins=40, kde=True)
    plt.title("Delay Rate Distribution")
    plt.xlabel("Delay rate (arr_del15 / arr_flights)")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = outdir / "plots" / "delay_rate_distribution.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_monthly_trend(df: pd.DataFrame, outdir: Path) -> Path:
    trend = (
        df.groupby("ym")
        .apply(lambda g: weighted_mean(g["delay_rate"], g["arr_flights"]))
        .reset_index(name="delay_rate_wavg")
        .sort_values("ym")
    )
    plt.figure(figsize=(10, 4))
    plt.plot(trend["ym"].astype(str), trend["delay_rate_wavg"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Weighted Average Delay Rate Over Time")
    plt.xlabel("YearMonth")
    plt.ylabel("Weighted delay rate")
    plt.tight_layout()
    out_path = outdir / "plots" / "trend_delay_rate.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_top_airports(df: pd.DataFrame, outdir: Path, top_n: int = 15, min_flights: int = 5000) -> Path:
    grp = df.groupby(["airport", "airport_name"]).agg(
        flights=("arr_flights", "sum"),
        delay_rate_wavg=("delay_rate", lambda s: weighted_mean(s, df.loc[s.index, "arr_flights"]))
    ).reset_index()
    grp = grp[grp["flights"] >= min_flights]
    top = grp.sort_values("delay_rate_wavg", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(y="airport_name", x="delay_rate_wavg", data=top, palette="Reds_r")
    plt.title(f"Top {len(top)} Airports by Weighted Delay Rate (>= {min_flights} flights)")
    plt.xlabel("Weighted delay rate")
    plt.ylabel("Airport")
    plt.tight_layout()
    out_path = outdir / "plots" / "top_airports_delay_rate.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_top_carriers(df: pd.DataFrame, outdir: Path, top_n: int = 15, min_flights: int = 5000) -> Path:
    grp = df.groupby(["carrier", "carrier_name"]).agg(
        flights=("arr_flights", "sum"),
        delay_rate_wavg=("delay_rate", lambda s: weighted_mean(s, df.loc[s.index, "arr_flights"]))
    ).reset_index()
    grp = grp[grp["flights"] >= min_flights]
    top = grp.sort_values("delay_rate_wavg", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(y="carrier_name", x="delay_rate_wavg", data=top, palette="Blues_r")
    plt.title(f"Top {len(top)} Carriers by Weighted Delay Rate (>= {min_flights} flights)")
    plt.xlabel("Weighted delay rate")
    plt.ylabel("Carrier")
    plt.tight_layout()
    out_path = outdir / "plots" / "top_carriers_delay_rate.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_month_carrier_heatmap(df: pd.DataFrame, outdir: Path, top_carriers: int = 10) -> Path:
    top_carrier_ids = (
        df.groupby("carrier")["arr_flights"].sum().sort_values(ascending=False).head(top_carriers).index
    )
    sub = df[df["carrier"].isin(top_carrier_ids)].copy()
    pivot = (
        sub.groupby(["carrier", "month"]).apply(lambda g: weighted_mean(g["delay_rate"], g["arr_flights"]))
        .reset_index(name="delay_rate")
    )
    heat = pivot.pivot(index="carrier", columns="month", values="delay_rate")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heat, annot=False, cmap="mako", vmin=0.0, vmax=min(0.6, float(np.nanmax(heat.values)) + 1e-6))
    plt.title("Delay Rate by Carrier (Top) and Month")
    plt.xlabel("Month")
    plt.ylabel("Carrier")
    plt.tight_layout()
    out_path = outdir / "plots" / "heatmap_carrier_month.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_cause_shares(df: pd.DataFrame, outdir: Path, top_carriers: int = 10, min_delayed: int = 1000) -> Path | None:
    cause_cols = ["carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct"]
    top_carrier_ids = (
        df.groupby("carrier")["arr_del15"].sum().sort_values(ascending=False).head(top_carriers).index
    )
    sub = df[df["carrier"].isin(top_carrier_ids)].copy()

    rows: List[Dict[str, float]] = []
    for carrier, g in sub.groupby("carrier"):
        delayed = g["arr_del15"].sum()
        if delayed < min_delayed:
            continue
        row: Dict[str, float | str] = {"carrier": carrier}
        for c in cause_cols:
            row[c] = g[c].sum() / delayed if delayed > 0 else float("nan")
        rows.append(row) 

    if not rows:
        return None

    plot_df = pd.DataFrame(rows).set_index("carrier")[cause_cols]

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(plot_df))
    labels = ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"]
    colors = sns.color_palette("Set2", n_colors=len(cause_cols))
    for idx, c in enumerate(cause_cols):
        plt.bar(plot_df.index, plot_df[c].values, bottom=bottom, label=labels[idx], color=colors[idx])
        bottom = bottom + plot_df[c].values
    plt.title("Cause Share of 15+ Min Delays by Carrier (Top)")
    plt.ylabel("Share of delayed flights")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Cause")
    plt.tight_layout()
    out_path = outdir / "plots" / "cause_shares_by_carrier.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def run_eda(df: pd.DataFrame, outdir: Path) -> List[Path]:
    paths: List[Path] = []
    paths.append(plot_delay_distribution(df, outdir))
    paths.append(plot_monthly_trend(df, outdir))
    paths.append(plot_top_airports(df, outdir))
    paths.append(plot_top_carriers(df, outdir))
    paths.append(plot_month_carrier_heatmap(df, outdir))
    p = plot_cause_shares(df, outdir)
    if p is not None:
        paths.append(p)
    return paths


# ----------------------------
# Simple target mean encoder
# ----------------------------

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Target mean encoder with smoothing to reduce leakage/noise.

    encoding = s * category_mean + (1 - s) * global_mean
    s = 1 / (1 + exp(-(n - min_samples_leaf) / smoothing))
    where n is the weight sum for that category.
    """

    def __init__(self, columns: List[str], min_samples_leaf: float = 50.0, smoothing: float = 10.0):
        self.columns = columns
        self.min_samples_leaf = float(min_samples_leaf)
        self.smoothing = float(smoothing)
        self.global_mean_: Optional[float] = None
        self.maps_: Dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        Xy = X.copy()
        Xy["_y"] = y.values
        Xy["_w"] = 1.0 if sample_weight is None else np.asarray(sample_weight, dtype=float)

        self.global_mean_ = float(np.average(Xy["_y"], weights=Xy["_w"]))

        for col in self.columns:
            grp = Xy.groupby(col).agg(
                wsum=("_w", "sum"),
                ysum=("_y", lambda s: np.sum(s.values * Xy.loc[s.index, "_w"].values)),
            )
            grp["mean"] = grp["ysum"] / grp["wsum"].replace(0.0, np.nan)
            grp["smoothing_factor"] = 1.0 / (1.0 + np.exp(-(grp["wsum"] - self.min_samples_leaf) / self.smoothing))
            grp["encoded"] = grp["smoothing_factor"] * grp["mean"] + (1.0 - grp["smoothing_factor"]) * self.global_mean_
            self.maps_[col] = grp["encoded"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("TargetMeanEncoder must be fit before transform.")
        X_out = pd.DataFrame(index=X.index)
        for col in self.columns:
            mapping = self.maps_.get(col, pd.Series(dtype=float))
            X_out[f"{col}_te"] = X[col].map(mapping).fillna(self.global_mean_).astype(float)
        return X_out


# ----------------------------
# Modeling
# ----------------------------

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["year"] = df["year"].astype(int)
    features["month_sin"] = df["month_sin"].astype(float)
    features["month_cos"] = df["month_cos"].astype(float)
    features["arr_flights_log1p"] = df["arr_flights_log1p"].astype(float)
    return features


def time_based_train_test_split(df: pd.DataFrame, test_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_ym = np.sort(df["ym"].unique())
    if len(unique_ym) <= test_months:
        test_months = max(1, len(unique_ym) // 3) or 1
    cutoff_values = unique_ym[-test_months:]
    train_df = df[~df["ym"].isin(cutoff_values)].copy()
    test_df = df[df["ym"].isin(cutoff_values)].copy()
    return train_df, test_df


def train_model(df: pd.DataFrame, outdir: Path, test_months: int = 6) -> Dict[str, float]:
    train_df, test_df = time_based_train_test_split(df, test_months=test_months)

    base_train = build_base_features(train_df)
    base_test = build_base_features(test_df)

    y_train = train_df["delay_rate"].astype(float)
    y_test = test_df["delay_rate"].astype(float)
    w_train = train_df["arr_flights"].astype(float).values
    w_test = test_df["arr_flights"].astype(float).values

    encoder = TargetMeanEncoder(columns=["carrier", "airport"], min_samples_leaf=200.0, smoothing=20.0)
    enc_train = encoder.fit(train_df[["carrier", "airport"]], y_train, sample_weight=w_train).transform(
        train_df[["carrier", "airport"]]
    )
    enc_test = encoder.transform(test_df[["carrier", "airport"]])

    X_train = pd.concat([base_train, enc_train], axis=1)
    X_test = pd.concat([base_test, enc_test], axis=1)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=None,
        max_iter=600,
        learning_rate=0.05,
        l2_regularization=0.0,
        max_bins=255,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    metrics: Dict[str, float | int | str] = {
        "test_mae": float(mean_absolute_error(y_test, y_pred, sample_weight=w_test)),
        "test_r2": float(r2_score(y_test, y_pred, sample_weight=w_test)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_months": int(len(np.unique(test_df["ym"]))),
    }

    model_path = outdir / "artifacts" / "model.pkl"
    joblib.dump({"model": model, "encoder": encoder, "feature_columns": list(X_train.columns)}, model_path)
    print(f"Saved model: {model_path}")

    try:
        pi = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error"
        )
        pi_df = pd.DataFrame(
            {"feature": X_test.columns, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}
        ).sort_values("importance_mean", ascending=False)
        pi_path = outdir / "artifacts" / "permutation_importance.csv"
        pi_df.to_csv(pi_path, index=False)
        print(f"Saved permutation importance: {pi_path}")

        plt.figure(figsize=(8, 6))
        sns.barplot(data=pi_df.head(15), x="importance_mean", y="feature", color="tab:blue")
        plt.title("Permutation Importance (top 15)")
        plt.xlabel("Mean importance (Δ MAE)")
        plt.ylabel("Feature")
        plt.tight_layout()
        pi_plot_path = outdir / "plots" / "permutation_importance.png"
        plt.savefig(pi_plot_path)
        plt.close()
        print(f"Saved permutation importance plot: {pi_plot_path}")
    except Exception as e:
        metrics["permutation_importance_error"] = str(e)
        print(f"Permutation importance failed: {e}")

    metrics_path = outdir / "artifacts" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    return metrics


# ----------------------------
# Summary stats
# ----------------------------

def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    overall = {
        "rows": int(len(df)),
        "date_min": int(df["ym"].min()),
        "date_max": int(df["ym"].max()),
        "flights_total": float(df["arr_flights"].sum()),
        "delayed_15_total": float(df["arr_del15"].sum()),
        "delay_rate_weighted": float(weighted_mean(df["delay_rate"], df["arr_flights"])),
        "cancel_rate_weighted": float(weighted_mean(df["cancel_rate"], df["arr_flights"])),
        "divert_rate_weighted": float(weighted_mean(df["divert_rate"], df["arr_flights"])),
        "avg_arr_delay_per_flight_weighted": float(weighted_mean(df["avg_arr_delay_per_flight"], df["arr_flights"])),
    }
    return overall


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Flight Delay Analysis (BTS aggregated data)")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--outdir", default="outputs", help="Directory to write outputs (default: outputs)")
    parser.add_argument("--test_months", type=int, default=6, help="Number of last months for test split")
    parser.add_argument("--skip_plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    print(f"Loading data: {input_path}")
    df_raw = load_data(input_path)
    print(f"Rows loaded: {len(df_raw):,}")

    print("Cleaning and engineering features...")
    df = clean_and_engineer(df_raw)
    cleaned_path = save_parquet_or_csv(df, outdir / "cleaned.parquet")
    print(f"Saved cleaned data: {cleaned_path}")

    if not args.skip_plots:
        print("Running EDA and saving plots...")
        sns.set_context("talk")
        plot_paths = run_eda(df, outdir)
        for p in plot_paths:
            print(f"Saved plot: {p}")
    else:
        print("Skipping plots as requested.")

    print("Computing summary stats...")
    summary = compute_summary(df)
    summary_path = outdir / "artifacts" / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary stats: {summary_path}")
    print(json.dumps(summary, indent=2))

    print("Training model (predict delay_rate)...")
    metrics = train_model(df, outdir, test_months=args.test_months)
    print("Metrics:\n" + json.dumps(metrics, indent=2))

    print(f"Done. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
