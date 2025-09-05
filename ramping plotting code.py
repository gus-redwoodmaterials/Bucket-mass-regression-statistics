from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import os
import sys

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download

# Configuration
DATA_FOLDER = "data"

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Using the working tag names from your SQL query
TAGS = {
    "feed_rate_main_fan": "rc1/4420-calciner/acmestatus/feed_rate_main_fan_limited",
    "feed_rate_requested": "rc1/4420-calciner/acmestatus/feed_rate_request",
    "feed_rate_target": "rc1/4420-calciner/acmestatus/feed_rate_target",
    # "bucket_mass": "rc1/4420-calciner/acmestatus/bucket_mass",
    # "bucket_mass_regress": "rc1/4420-calciner/acmestatus/bucket_mass_regress_c_large_mod_mass",
    # "bucket_mass_rpm_const": "rc1/4420-calciner/acmestatus/bucket_mass_regress_c_rpm_const",
    # "bucket_mass_resid": "rc1/4420-calciner/acmestatus/bucket_mass_regress_resid",
    # "infeed_hz_sp": "rc1/4420-calciner/acmestatus/hz_infeed_convey_mpc_ts_now",
    # "large_module_cycle_time": "rc1/4420-calciner/acmestatus/module_cycle_time_mpc_ts_now",
    # "cycle_time_complete": "rc1/4420-calciner/module_loading/hmi/cycle_time_complete/value",
    # "infeed_hz_actual": "rc1/4420-calciner/4420-cvr-001/status/speed_feedback_hz",
    # "kiln_weight_avg": "RC1/4420-Calciner/HMI/KilnWeightAvg/Value".lower(),
    # "kiln_rpm": "rc1/4420-calciner/4420-kln-001_rpm/input",
    # "kiln_weight_setpoint": "rc1/4420-calciner/hmi/4420-kln-001_weight_sp/value",
    # "acme_alive": "rc1/4420-calciner/acme/acmefeedrate/acmealive",
    "acme_engineering": "rc1/4420-calciner/acme/acmefeedrate/acmeengineeringenable",
    # "feed_robot_mode": "rc1/plc8-infeed_robot/hmi/feed_robot_mode/value",
    # "main_fan_speed": "rc1/4430-exhaust/4430-fan-004/status/speed_feedback_hz",
    # "startup_on": "rc1/4420-calciner/acme/acmefeedrate/kiln_bed_build_mode_honor_request",
    # "tipper_one_on": "RC1/4410-FeedPrep/4410-VFR-001/Status/Speed_Feedback_Hz".lower(),
    # "tipper_two_on": "RC1/4410-FeedPrep/4410-VFR-002/Status/Speed_Feedback_Hz".lower(),
}

# Database configuration
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")
DATA_TIMESTEP_SECONDS = 3


def load_data(start_date_str, end_date_str, description="data"):
    """
    Load data for a specified date range

    Args:
        start_date_str: Start date in format "2025-07-20T08:32:00"
        end_date_str: End date in format "2025-07-21T12:00:00"
        description: Description for the dataset (e.g., "before_controller", "after_controller")

    Returns:
        DataFrame with the loaded data
    """

    # Parse naive date strings as Pacific time, then convert to UTC for Athena
    start_pacific = pacific_tz.localize(parse(start_date_str))
    end_pacific = pacific_tz.localize(parse(end_date_str))
    start_utc = start_pacific.astimezone(pytz.utc)
    end_utc = end_pacific.astimezone(pytz.utc)

    # Generate filename based on Pacific time (for user clarity)
    start_str = start_pacific.strftime("%Y%m%d_%H%M")
    end_str = end_pacific.strftime("%Y%m%d_%H%M")
    csv_filename = f"feedrate_analysis_{description}_{start_str}_to_{end_str}.csv"
    csv_path = os.path.join(DATA_FOLDER, csv_filename)

    print(
        f"\n📊 Loading {description} data: {start_pacific.strftime('%m/%d %H:%M')} to {end_pacific.strftime('%m/%d %H:%M')} (Pacific)"
    )

    # Check command line arguments for refresh flag
    refresh_data = "-refresh" in sys.argv or "-r" in sys.argv

    # Try to load from cache first
    if not refresh_data and os.path.exists(csv_path):
        print("   Loading from cache...")
        df = pd.read_csv(csv_path)
        # Convert timestamp back to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        print("   Fetching from Athena...")
        df = athena_download.get_pivoted_athena_data(
            TAGS,
            start_utc,
            end_utc,
            TABLE_PATH,
            DATA_TIMESTEP_SECONDS,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Save to CSV for future use
        df.to_csv(csv_path, index=False)
        print("   Cached for future use")

    # Only look at times where acme is running without the robot
    df = df[df["acme_engineering"] == 1.0].copy()

    filtered_rows = len(df)
    print(f"   Filtered to {filtered_rows} rows where ACME engineering is enabled")
    return df


def plot_ramping(df, title="Feed Rate Ramping Analysis"):
    # Use a small tolerance for floating point comparison
    tolerance = 0.001
    equal_mask = abs(df["feed_rate_target"] - df["feed_rate_main_fan"]) < tolerance

    # Find the start and end points of each equal period
    changes = equal_mask.astype(int).diff().fillna(0)
    start_idx = df.index[changes == 1]
    end_idx = df.index[changes == -1]

    # Handle case where the first period starts at the beginning
    if equal_mask.iloc[0]:
        start_idx = pd.Index([df.index[0]]).append(start_idx)
    # Handle case where the last period ends at the end
    if equal_mask.iloc[-1]:
        end_idx = end_idx.append(pd.Index([df.index[-1]]))

    # Filter periods that last more than one hour
    long_periods = []
    print("\nPeriods where feed rates were equal:")
    for start, end in zip(start_idx, end_idx):
        start_time = df.loc[start, "timestamp"]
        end_time = df.loc[end, "timestamp"]
        duration = (end_time - start_time).total_seconds() / 3600  # Convert to hours
        if duration >= 1:
            long_periods.append((start, end, duration))
            print(
                f"   Period {len(long_periods)}: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} ({duration:.1f} hours)"
            )

    if not long_periods:
        print("   No periods found where feed rates were equal for more than one hour")
        return

    # Plot feed rates for each period in its own figure
    for idx, (start, end, duration) in enumerate(long_periods):
        # Create a new figure for each period
        plt.figure(figsize=(15, 8))

        mask = (df["timestamp"] >= df.loc[start, "timestamp"]) & (df["timestamp"] <= df.loc[end, "timestamp"])
        period_df = df[mask]

        # Plot this period
        plt.plot(
            period_df["timestamp"],
            period_df["feed_rate_requested"],
            label="Requested Feed Rate",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            period_df["timestamp"],
            period_df["feed_rate_target"],
            label="Target Feed Rate",
            color="orange",
            linewidth=2,
        )

        # Set labels and grid
        plt.xlabel("Time")
        plt.ylabel("Feed Rate (units)")
        period_title = f"Period {idx + 1}: {df.loc[start, 'timestamp'].strftime('%Y-%m-%d %H:%M')} to {df.loc[end, 'timestamp'].strftime('%Y-%m-%d %H:%M')}\nDuration: {duration:.1f} hours"
        plt.title(period_title)
        plt.grid(True)
        plt.legend()

        # Rotate x-axis labels for better readability
        plt.tick_params(axis="x", rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

    # Show all figures
    plt.show()


def run():
    # Example usage
    df = load_data("2025-08-24T08:32:00", "2025-08-27T12:00:00", description="example")
    print(df.head())
    plot_ramping(df, title="Feed Rate Ramping Analysis")


if __name__ == "__main__":
    run()
