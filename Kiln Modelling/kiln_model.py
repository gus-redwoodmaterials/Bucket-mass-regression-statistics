import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import os
import sys
from datetime import datetime, timedelta
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download
import warnings
from scipy.optimize import curve_fit

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

MODEL_EXPONENT = 1 / 2  # Change this to 0.5 for square root model, 1 for linear model

# Configuration
DATA_FOLDER = "/Users/gus.robinson/Desktop/Local Python Coding/Kiln Modelling/Data"

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Tags for kiln weight and RPM
TAGS = {
    "kiln_weight_avg": "rc1/4420-calciner/hmi/kilnweightavg/value",
    "kiln_rpm": "rc1/4420-calciner/4420-kln-001_rpm/input",
    "mass_flow_infeed_convey": "rc1/4420-calciner/acmestatus/mass_flow_infeed_convey_rolling_average",
    "mass_flow_module_convey": "rc1/4420-calciner/acmestatus/mass_flow_module_convey_rolling_average",
}

SOLIDS_MASS_FLOW_CONVERSION = 0.7


# Database configuration
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")
DATA_TIMESTEP_SECONDS = 3


def load_kiln_data(start_date_str, end_date_str, description="kiln_data"):
    """
    Load kiln weight and RPM data for a specified date range

    Args:
        start_date_str: Start date in format "2025-07-20T08:32:00"
        end_date_str: End date in format "2025-07-21T12:00:00"
        description: Description for the dataset

    Returns:
        DataFrame with the loaded data
    """

    # Parse dates and localize to Pacific timezone
    start = pacific_tz.localize(parse(start_date_str))
    end = pacific_tz.localize(parse(end_date_str))

    # Generate filename based on date range and description
    start_str = start.strftime("%Y%m%d_%H%M")
    end_str = end.strftime("%Y%m%d_%H%M")
    csv_filename = f"{description}_{start_str}_to_{end_str}.csv"
    csv_path = os.path.join(DATA_FOLDER, csv_filename)

    print(f"\n📊 Loading {description} data: {start.strftime('%m/%d %H:%M')} to {end.strftime('%m/%d %H:%M')}")

    # Check command line arguments for refresh flag
    refresh_data = "--refresh" in sys.argv or "-r" in sys.argv

    # Try to load from cache first
    if not refresh_data and os.path.exists(csv_path):
        print(f"   Loading from cache...")
        df = pd.read_csv(csv_path)
        # Convert timestamp back to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        print("   Fetching from Athena...")
        df = athena_download.get_pivoted_athena_data(
            TAGS,
            start,
            end,
            TABLE_PATH,
            DATA_TIMESTEP_SECONDS,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(pacific_tz)
        # Save to CSV for future use
        df.to_csv(csv_path, index=False)
        print("   Cached for future use")

    df["mass_inflow_feed"] = df["mass_flow_infeed_convey"] + df["mass_flow_module_convey"]

    print(df.head())  # Print first few rows for verification
    return df


def create_analysis_dataframe(kiln_df, ms4_df, time_start=None, time_end=None, window_minutes=30):
    # Ensure datetime columns
    kiln_df["timestamp"] = pd.to_datetime(kiln_df["timestamp"], utc=True)
    ms4_df["start_time_utc"] = pd.to_datetime(ms4_df["start_time_utc"], utc=True)
    ms4_df["end_time_utc"] = pd.to_datetime(ms4_df["end_time_utc"], utc=True)

    # Set time range
    if time_start is None:
        time_start = max(kiln_df["timestamp"].min(), ms4_df["start_time_utc"].min())
    if time_end is None:
        time_end = min(kiln_df["timestamp"].max(), ms4_df["end_time_utc"].max())

    # Generate minute-by-minute timestamps
    window_starts = pd.date_range(start=time_start, end=time_end, freq=f"{window_minutes}T", tz="UTC")

    results = []
    for window_start in window_starts:
        window_end = window_start + pd.Timedelta(minutes=window_minutes)

        # Get kiln data for this window
        kiln_window = kiln_df[(kiln_df["timestamp"] >= window_start) & (kiln_df["timestamp"] < window_end)]
        if len(kiln_window) == 0:
            continue

        # Average kiln parameters over the window
        kiln_weight_avg = kiln_window["kiln_weight_avg"].mean()
        kiln_rpm_avg = kiln_window["kiln_rpm"].mean()
        mass_inflow_feed_avg = kiln_window["mass_inflow_feed"].mean()

        # Get MS4 events COMPLETED within this window (use end_time_utc)
        # This captures all material that came out during this window
        ms4_window = ms4_df[(ms4_df["end_time_utc"] >= window_start) & (ms4_df["end_time_utc"] < window_end)]
        total_mass_out = ms4_window["net_weight"].sum() if not ms4_window.empty else 0

        # Calculate mass flow rate in kg/hr
        hours = window_minutes / 60.0
        mass_flow_rate = total_mass_out / hours if hours > 0 else 0

        results.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "kiln_weight_avg": kiln_weight_avg,
                "kiln_rpm_avg": kiln_rpm_avg,
                "mass_inflow_feed_avg": mass_inflow_feed_avg,
                "total_mass_out": total_mass_out,
                "mass_flow_rate_kg_hr": mass_flow_rate,
            }
        )

    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.dropna(subset=["kiln_weight_avg", "kiln_rpm_avg", "mass_inflow_feed_avg"])
    return analysis_df


def fit_k_constant(df):
    """
    Fit the constant k for the model: net_weight = k * kiln_weight_avg * kiln_rpm_avg
    Returns the fitted k value.
    """
    # Drop rows with missing or zero values to avoid log(0)
    df = df.dropna(subset=["kiln_weight_avg", "kiln_rpm_avg"])
    df = df[(df["kiln_weight_avg"] > 0) & (df["kiln_rpm_avg"] > 0)]

    # Model function
    def model(X, k):
        kiln_weight, kiln_rpm = X
        return k * kiln_weight ** (MODEL_EXPONENT) * kiln_rpm

    # regressing against our rate of weight out (kg/s) instead of total net weight
    X = np.vstack([df["kiln_weight_avg"], df["kiln_rpm_avg"]])
    y = df["mass_flow_rate_kg_hr"].values

    # Fit k using curve_fit
    popt, _ = curve_fit(model, X, y)
    k_fit = popt[0]
    print(f"Fitted k: {k_fit:.4f}")
    return k_fit


def fit_k_and_v_constants(df):
    """
    Fit constants k and v for the model: mass_flow_rate_kg_hr = k * kiln_weight_avg ** v * kiln_rpm_avg
    Returns the fitted k and v values.
    """
    # Drop rows with missing or zero values to avoid log(0)
    df = df.dropna(subset=["kiln_weight_avg", "kiln_rpm_avg"])
    df = df[(df["kiln_weight_avg"] > 0) & (df["kiln_rpm_avg"] > 0)]

    # Model function
    def model(X, k, v):
        kiln_weight, kiln_rpm = X
        return k * kiln_weight**v * kiln_rpm

    X = np.vstack([df["kiln_weight_avg"], df["kiln_rpm_avg"]])
    y = df["mass_flow_rate_kg_hr"].values

    # Only bound v (0.25 to 4), leave k completely unbounded
    popt, _ = curve_fit(model, X, y, bounds=([-np.inf, 0.25], [np.inf, 4]))
    k_fit, v_fit = popt
    print(f"Fitted k: {k_fit:.4f}, Fitted v: {v_fit:.4f}")
    return k_fit, v_fit


def evaluate_model(analysis_df, ms4_df, k_fit, v_fit, start_time, end_time):
    """
    Evaluate the model performance by comparing actual total mass out vs predicted total mass out.

    Args:
        analysis_df: DataFrame with window_start, window_end, kiln variables, mass_flow_rate_kg_hr
        ms4_df: DataFrame with start_time_utc, end_time_utc, net_weight
        k_fit: Fitted k constant for the model
        start_time, end_time: Time range for evaluation
        model_exponent: Exponent for kiln_weight_avg in the model (default 0.5 for square root)
    """
    # Filter DataFrames to the specified time range
    analysis_window = analysis_df[
        (analysis_df["window_start"] >= start_time) & (analysis_df["window_end"] <= end_time)
    ].copy()
    ms4_window = ms4_df[(ms4_df["end_time_utc"] >= start_time) & (ms4_df["end_time_utc"] <= end_time)]

    # Actual total mass out from MS4 data
    actual_total_mass = ms4_window["net_weight"].sum()

    # Predict mass flow rate for each window using our model
    for i, row in analysis_window.iterrows():
        # Calculate predicted mass flow rate using our model
        predicted_flow_rate = k_fit * (row["kiln_weight_avg"] ** v_fit) * row["kiln_rpm_avg"]
        analysis_window.at[i, "predicted_flow_rate_kg_hr"] = predicted_flow_rate

        # Calculate window duration in hours
        window_duration_hours = (row["window_end"] - row["window_start"]).total_seconds() / 3600

        # Calculate predicted mass out for this window (integral of flow rate * time)
        predicted_mass_out = predicted_flow_rate * window_duration_hours
        analysis_window.at[i, "predicted_mass_out"] = predicted_mass_out

    # Total predicted mass out (sum of all windows)
    predicted_total_mass = analysis_window["predicted_mass_out"].sum()

    # Calculate error metrics
    absolute_error = abs(actual_total_mass - predicted_total_mass)
    relative_error = absolute_error / actual_total_mass * 100 if actual_total_mass > 0 else float("inf")

    # Calculate RMSE for flow rate predictions vs actual flow rates
    analysis_window = analysis_window.dropna(subset=["mass_flow_rate_kg_hr"])
    if len(analysis_window) > 0:
        residuals = analysis_window["mass_flow_rate_kg_hr"] - analysis_window["predicted_flow_rate_kg_hr"]
        rmse = np.sqrt(np.mean(residuals**2))

        # Calculate R-squared
        ss_total = np.sum(
            (analysis_window["mass_flow_rate_kg_hr"] - analysis_window["mass_flow_rate_kg_hr"].mean()) ** 2
        )
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    else:
        rmse = None
        r_squared = None

    # Print results
    print(f"\nModel Evaluation from {start_time} to {end_time}:")
    print(f"  Actual total mass out: {actual_total_mass:.2f} kg")
    print(f"  Predicted total mass out: {predicted_total_mass:.2f} kg")
    print(f"  Absolute error: {absolute_error:.2f} kg")
    print(f"  Relative error: {relative_error:.2f}%")
    if rmse is not None:
        print(f"  RMSE (flow rate): {rmse:.2f} kg/hr")
        print(f"  R² (flow rate): {r_squared:.4f}")

    # Create a plot comparing actual vs predicted flow rates
    if len(analysis_window) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(
            analysis_window["window_start"],
            analysis_window["mass_flow_rate_kg_hr"],
            label="Actual Flow Rate",
            marker="o",
        )
        plt.plot(
            analysis_window["window_start"],
            analysis_window["predicted_flow_rate_kg_hr"],
            label="Predicted Flow Rate",
            marker="x",
        )
        plt.xlabel("Time")
        plt.ylabel("Mass Flow Rate (kg/hr)")
        plt.title("Actual vs Predicted Mass Flow Rate")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return {
        "actual_total_mass": actual_total_mass,
        "predicted_total_mass": predicted_total_mass,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "rmse": rmse,
        "r_squared": r_squared,
        "analysis_window": analysis_window,  # Return the window for further analysis
    }


def run_analysis(kiln_df, ms4_df, start_time=None, end_time=None):
    analysis_df = create_analysis_dataframe(kiln_df, ms4_df, time_start=start_time, time_end=end_time)
    print("\nAnalysis DataFrame:")
    print(analysis_df.head())
    k_fit = fit_k_constant(analysis_df)
    # k_fit, v_fit = fit_k_and_v_constants(analysis_df)
    # plot_kiln_relationship(analysis_df, k_fit)
    return k_fit, analysis_df


def plot_data_frames(kiln_data_df, ms4_out_df, start_time=None, end_time=None):
    if start_time is not None and end_time is not None:
        kiln_data_df = kiln_data_df[(kiln_data_df["timestamp"] >= start_time) & (kiln_data_df["timestamp"] <= end_time)]
        ms4_out_df = ms4_out_df[(ms4_out_df["start_time_utc"] >= start_time) & (ms4_out_df["end_time_utc"] <= end_time)]

    # Create a figure with two vertically stacked subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top subplot: kiln_weight_avg and kiln_rpm
    ax1.plot(kiln_data_df["timestamp"], kiln_data_df["kiln_weight_avg"], label="Kiln Weight", color="orange")
    ax1.set_ylabel("Kiln Weight", color="orange")
    ax1.tick_params(axis="y", labelcolor="orange")
    ax1.set_title("Kiln Weight and Kiln RPM Over Time")

    ax2 = ax1.twinx()
    ax2.plot(kiln_data_df["timestamp"], kiln_data_df["kiln_rpm"], label="Kiln RPM", color="blue", alpha=0.6)
    ax2.set_ylabel("Kiln RPM", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Bottom subplot: MS4 net weight
    ms4_out_df["start_time_utc"] = pd.to_datetime(ms4_out_df["start_time_utc"], utc=True)
    ax3.scatter(ms4_out_df["start_time_utc"], ms4_out_df["net_weight"], label="MS4 Net Weight", color="orange", s=10)
    ax3.set_ylabel("MS4 Net Weight", color="orange")
    ax3.set_xlabel("Time")
    ax3.set_title("MS4 Net Weight Over Time")
    ax3.tick_params(axis="y", labelcolor="orange")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


def run():
    kiln_data_df = load_kiln_data("'2025-06-01 00:00:00'", "'2025-09-08 00:00:00'", description="kiln_data")
    ms4_out_df = pd.read_csv("/Users/gus.robinson/Desktop/Local Python Coding/Kiln Modelling/Data/ms4_out_data.csv")
    ms4_out_df = ms4_out_df[ms4_out_df["net_weight"] > 1].copy()  # Filter out very small weights

    # After loading kiln_df and ms4_df from CSV, ensure all timestamps are UTC and timezone-aware
    kiln_data_df["timestamp"] = pd.to_datetime(kiln_data_df["timestamp"], utc=True)
    if "start_time_utc" in ms4_out_df.columns:
        ms4_out_df["start_time_utc"] = pd.to_datetime(ms4_out_df["start_time_utc"], utc=True)
    if "end_time_utc" in ms4_out_df.columns:
        ms4_out_df["end_time_utc"] = pd.to_datetime(ms4_out_df["end_time_utc"], utc=True)

    analysis_df_start_time = pd.Timestamp("2025-08-28 00:00:00", tz="UTC")
    analysis_df_end_time = pd.Timestamp("2025-09-03 00:00:00", tz="UTC")
    k_fit, analysis_df = run_analysis(
        kiln_data_df, ms4_out_df, start_time=analysis_df_start_time, end_time=analysis_df_end_time
    )
    evaluation_start_time = pd.Timestamp("2025-08-26 00:00:00", tz="UTC")
    evaluation_end_time = pd.Timestamp("2025-08-28 00:00:00", tz="UTC")
    helper_analysis_df = create_analysis_dataframe(
        kiln_data_df, ms4_out_df, time_start=evaluation_start_time, time_end=evaluation_end_time
    )
    evaluate_model(helper_analysis_df, ms4_out_df, k_fit, MODEL_EXPONENT, evaluation_start_time, evaluation_end_time)

    # plot_data_frames(kiln_data_df, ms4_out_df, start_time=start_time, end_time=end_time)


if __name__ == "__main__":
    run()
