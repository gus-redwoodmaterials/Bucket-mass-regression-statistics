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
    minute_range = pd.date_range(start=time_start, end=time_end, freq="1T", tz="UTC")

    results = []
    for t in minute_range:
        print(f"Processing time: {t}", end="\r")
        # Rolling window for kiln data
        t_start = t - pd.Timedelta(minutes=window_minutes)
        kiln_window = kiln_df[(kiln_df["timestamp"] >= t_start) & (kiln_df["timestamp"] <= t)]
        kiln_weight_avg = kiln_window["kiln_weight_avg"].mean()
        kiln_rpm_avg = kiln_window["kiln_rpm"].mean()
        mass_inflow_feed_avg = kiln_window["mass_inflow_feed"].mean()

        # Find ms4 output that matches this minute (optional: nearest or within window)
        ms4_row = ms4_df[(ms4_df["start_time_utc"] <= t) & (ms4_df["end_time_utc"] >= t)]
        net_weight = ms4_row["net_weight"].mean() if not ms4_row.empty else np.nan

        results.append(
            {
                "timestamp": t,
                "kiln_weight_avg": kiln_weight_avg,
                "kiln_rpm_avg": kiln_rpm_avg,
                "mass_inflow_feed_avg": mass_inflow_feed_avg,
                "net_weight": net_weight,
            }
        )

    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.dropna(subset=["kiln_weight_avg", "kiln_rpm_avg", "mass_inflow_feed_avg"])
    return analysis_df


def compute_rolling_averages(kiln_df, ms4_df, window_minutes=30):
    """
    For each ms4 transaction_time, compute rolling averages of kiln variables over the preceding window_minutes.
    Returns a DataFrame with rolling averages aligned to ms4 transaction times.
    """
    # Ensure datetime columns
    kiln_df["timestamp"] = pd.to_datetime(kiln_df["timestamp"], utc=True)
    ms4_df["transaction_time_utc"] = pd.to_datetime(ms4_df["transaction_time_utc"], utc=True)

    results = []
    for _, row in ms4_df.iterrows():
        t_end = row["transaction_time_utc"]
        t_start = t_end - pd.Timedelta(minutes=window_minutes)
        mask = (kiln_df["timestamp"] >= t_start) & (kiln_df["timestamp"] <= t_end)
        kiln_window = kiln_df.loc[mask]
        results.append(
            {
                "transaction_time_utc": t_end,
                "kiln_weight_avg": kiln_window["kiln_weight_avg"].mean(),
                "kiln_rpm_avg": kiln_window["kiln_rpm"].mean(),
                "mass_inflow_feed_avg": kiln_window["mass_inflow_feed"].mean(),
                "net_weight": row.get("net_weight", None),
            }
        )
    rolling_df = pd.DataFrame(results)
    rolling_df = rolling_df.dropna()
    return rolling_df


def fit_k_constant(df):
    """
    Fit the constant k for the model: net_weight = k * kiln_weight_avg * kiln_rpm_avg
    Returns the fitted k value.
    """
    # Drop rows with missing or zero values to avoid log(0)
    df = df.dropna(subset=["net_weight", "kiln_weight_avg", "kiln_rpm_avg"])
    df = df[(df["net_weight"] > 0) & (df["kiln_weight_avg"] > 0) & (df["kiln_rpm_avg"] > 0)]

    # Model function
    def model(X, k):
        kiln_weight, kiln_rpm = X
        return k * kiln_weight ** (1 / 2) * kiln_rpm

    X = np.vstack([df["kiln_weight_avg"], df["kiln_rpm_avg"]])
    y = df["net_weight"].values

    # Fit k using curve_fit
    popt, _ = curve_fit(model, X, y)
    k_fit = popt[0]
    print(f"Fitted k: {k_fit:.4f}")
    return k_fit


def plot_kiln_relationship(df, k_fit):
    """
    Plot actual net_weight vs modeled net_weight using fitted k.
    """
    df = df.dropna(subset=["net_weight", "kiln_weight_avg", "kiln_rpm_avg"])
    modeled = k_fit * df["kiln_weight_avg"] * df["kiln_rpm_avg"]

    plt.figure(figsize=(8, 6))
    plt.scatter(df["net_weight"], modeled, alpha=0.7)
    # plt.plot(
    #     [df["net_weight"].min(), df["net_weight"].max()],
    #     [df["net_weight"].min(), df["net_weight"].max()],
    #     "r--",
    #     label="Ideal Fit",
    # )
    plt.xlabel("Actual Net Weight")
    plt.ylabel("Modeled Net Weight")
    plt.title("Actual vs Modeled Net Weight (k * kiln_weight_avg * kiln_rpm_avg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_analysis(kiln_df, ms4_df, start_time=None, end_time=None):
    analysis_df = create_analysis_dataframe(kiln_df, ms4_df, time_start=start_time, time_end=end_time)
    k_fit = fit_k_constant(analysis_df)
    plot_kiln_relationship(analysis_df, k_fit)
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

    start_time = pd.Timestamp("2025-08-28 00:00:00", tz="UTC")
    end_time = pd.Timestamp("2025-09-03 00:00:00", tz="UTC")
    # run_analysis(kiln_data_df, ms4_out_df, start_time=start_time, end_time=end_time)

    plot_data_frames(kiln_data_df, ms4_out_df, start_time=start_time, end_time=end_time)


if __name__ == "__main__":
    run()
