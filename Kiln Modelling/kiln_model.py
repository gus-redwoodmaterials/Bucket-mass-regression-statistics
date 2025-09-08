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


def create_analysis_dataframe(kiln_df, ms4_df):
    # Ensure datetime columns
    kiln_df["timestamp"] = pd.to_datetime(kiln_df["timestamp"])
    ms4_df["start_time_utc"] = pd.to_datetime(ms4_df["start_time_utc"])
    ms4_df["end_time_utc"] = pd.to_datetime(ms4_df["end_time_utc"])

    # Prepare lists to collect results
    results = []

    # For each MS4 output, aggregate kiln data over its time window
    for _, row in ms4_df.iterrows():
        mask = (kiln_df["timestamp"] >= row["start_time_utc"]) & (kiln_df["timestamp"] <= row["end_time_utc"])
        kiln_window = kiln_df.loc[mask]

        # Compute averages (or other stats) for kiln variables in this window
        kiln_weight_avg = kiln_window["kiln_weight_avg"].mean()
        kiln_rpm_avg = kiln_window["kiln_rpm"].mean()
        mass_inflow_feed_avg = kiln_window["mass_inflow_feed"].mean()

        # Build combined row
        results.append(
            {
                "kiln_weight_avg": kiln_weight_avg,
                "kiln_rpm_avg": kiln_rpm_avg,
                "mass_inflow_feed_avg": mass_inflow_feed_avg,
                "net_weight": row["net_weight"],
                "start_time_utc": row["start_time_utc"],
                "end_time_utc": row["end_time_utc"],
            }
        )

    # Create DataFrame
    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.dropna()

    return analysis_df


def kiln_model_solids_out(df):
    """
    Simple kiln model to estimate weight based on RPM and mass flow

    Args:
        df: DataFrame with kiln data

    Returns:
        DataFrame with actual and modeled kiln weight
    """

    # Constants for the model
    KILN_VOLUME = 10.0  # m^3, example volume of the kiln
    DENSITY_SOLIDS = 1.5  # tonnes/m^3, example density of solids in the kiln

    # Calculate the expected weight based on RPM and mass flow
    df["modeled_kiln_weight"] = (df["kiln_rpm"] / 100.0) * KILN_VOLUME * DENSITY_SOLIDS + df["mass_inflow_feed"]

    return df


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
        return k * kiln_weight * kiln_rpm

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
    plt.plot(
        [df["net_weight"].min(), df["net_weight"].max()],
        [df["net_weight"].min(), df["net_weight"].max()],
        "r--",
        label="Ideal Fit",
    )
    plt.xlabel("Actual Net Weight")
    plt.ylabel("Modeled Net Weight")
    plt.title("Actual vs Modeled Net Weight (k * kiln_weight_avg * kiln_rpm_avg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_analysis(kiln_df, ms4_df):
    analysis_df = create_analysis_dataframe(kiln_df, ms4_df)
    k_fit = fit_k_constant(analysis_df)
    plot_kiln_relationship(analysis_df, k_fit)
    return k_fit, analysis_df


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

    run_analysis(kiln_data_df, ms4_out_df)


if __name__ == "__main__":
    run()
