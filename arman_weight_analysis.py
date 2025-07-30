import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import os
import sys
from datetime import datetime, timedelta
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download

# Configuration
DATA_FOLDER = "data"

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Tags for kiln weight and RPM
TAGS = {
    "kiln_weight_avg": "rc1/4420-calciner/hmi/kilnweightavg/value",
    "kiln_rpm": "rc1/4420-calciner/4420-kln-001_rpm/input",
    "large_mod_feedrate": "rc1/4420-calciner/module_loading/hmi/cycle_time_complete/value",
    "small_mod_feedrate": "rc1/4420-calciner/4420-cvr-001/status/speed_feedback_hz",
}

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
    csv_filename = f"arman_weight_analysis_{description}_{start_str}_to_{end_str}.csv"
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

    return df


def plot_kiln_data(df, title_suffix=""):
    """
    Plot kiln weight, RPM, and feedrate data

    Args:
        df: DataFrame with all columns
        title_suffix: Optional suffix for plot titles
    """
    if df.empty:
        print("   ❌ No data to plot")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Plot 1: Kiln Weight
    if "kiln_weight_avg" in df.columns:
        ax1.plot(df["timestamp"], df["kiln_weight_avg"], color="blue", linewidth=1.5, label="Kiln Weight")
        ax1.set_ylabel("Kiln Weight (kg)")
        ax1.set_title(f"Kiln Weight Over Time{title_suffix}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Print basic statistics
        weight_stats = df["kiln_weight_avg"].describe()
        print(f"\n   📊 Kiln Weight Statistics:")
        print(f"   Mean: {weight_stats['mean']:.1f} kg")
        print(f"   Min: {weight_stats['min']:.1f} kg")
        print(f"   Max: {weight_stats['max']:.1f} kg")
        print(f"   Std: {weight_stats['std']:.1f} kg")

    # Plot 2: Kiln RPM
    if "kiln_rpm" in df.columns:
        ax2.plot(df["timestamp"], df["kiln_rpm"], color="green", linewidth=1.5, label="Kiln RPM")
        ax2.set_ylabel("Kiln RPM")
        ax2.set_title(f"Kiln RPM Over Time{title_suffix}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Print basic statistics
        rpm_stats = df["kiln_rpm"].describe()
        print(f"\n   📊 Kiln RPM Statistics:")
        print(f"   Mean: {rpm_stats['mean']:.2f} RPM")
        print(f"   Min: {rpm_stats['min']:.2f} RPM")
        print(f"   Max: {rpm_stats['max']:.2f} RPM")
        print(f"   Std: {rpm_stats['std']:.2f} RPM")

    # Plot 3: Large Module Feedrate
    if "large_mod_feedrate" in df.columns:
        ax3.plot(df["timestamp"], df["large_mod_feedrate"], color="red", linewidth=1.5, label="Large Mod Feedrate")
        ax3.set_ylabel("Large Mod Feedrate")
        ax3.set_title(f"Large Module Feedrate Over Time{title_suffix}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Print basic statistics
        large_stats = df["large_mod_feedrate"].describe()
        print(f"\n   📊 Large Module Feedrate Statistics:")
        print(f"   Mean: {large_stats['mean']:.2f}")
        print(f"   Min: {large_stats['min']:.2f}")
        print(f"   Max: {large_stats['max']:.2f}")
        print(f"   Std: {large_stats['std']:.2f}")

    # Plot 4: Small Module Feedrate
    if "small_mod_feedrate" in df.columns:
        ax4.plot(df["timestamp"], df["small_mod_feedrate"], color="orange", linewidth=1.5, label="Small Mod Feedrate")
        ax4.set_ylabel("Small Mod Feedrate")
        ax4.set_xlabel("Time")
        ax4.set_title(f"Small Module Feedrate Over Time{title_suffix}")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Print basic statistics
        small_stats = df["small_mod_feedrate"].describe()
        print(f"\n   📊 Small Module Feedrate Statistics:")
        print(f"   Mean: {small_stats['mean']:.2f}")
        print(f"   Min: {small_stats['min']:.2f}")
        print(f"   Max: {small_stats['max']:.2f}")
        print(f"   Std: {small_stats['std']:.2f}")

    # Format x-axis
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_segmented_kiln_data(df, coefficients=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(df["timestamp"], df["kiln_weight_avg"], "b-", linewidth=1, alpha=0.7)
    ax1.set_ylabel("Kiln Weight (kg)")
    ax1.set_title("Segmented Kiln Weight Data")
    ax1.grid(True, alpha=0.3)

    if coefficients is not None:
        # Convert timestamps to seconds for fitting
        delta_t = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        time_seconds = np.linspace(0, delta_t, len(df))

        # Create the polynomial function using the coefficients
        polynomial_function = np.poly1d(coefficients)

        # Generate fitted values
        y_fit = polynomial_function(time_seconds)

        # Plot the fitted line
        ax1.plot(
            df["timestamp"], y_fit, color="red", linewidth=2, label=f"Fit: slope = {coefficients[0] * 3600:.6f} kg/h"
        )
        ax1.legend()

    ax2.plot(df["timestamp"], df["kiln_rpm"], "r-", linewidth=1, alpha=0.7)
    ax2.set_ylabel("Kiln RPM")
    ax2.set_xlabel("Time (Pacific)")
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def solve_k_RPM(df):
    # filter out rows where small_mod_feedrate is 0 and large_mod_feedrate is constant
    filtered_df = df.loc[
        (df["small_mod_feedrate"] <= 0) & (df["large_mod_feedrate"] == df["large_mod_feedrate"].shift(1))
    ]

    # Create segments where data is NOT continuous (breaks in your condition)
    # This creates a new group ID every time the condition changes
    filtered_df = filtered_df.copy()
    filtered_df["segment"] = (~filtered_df.index.to_series().diff().eq(1)).cumsum()

    # Split into list of DataFrames
    segment_dfs = [group for name, group in filtered_df.groupby("segment")]

    ks = []
    rpms = []
    helper = []
    for segment in segment_dfs:
        if segment[["kiln_weight_avg", "kiln_rpm"]].isna().any().any() or len(segment) < 1000:
            continue

        end_time = segment["timestamp"].iloc[-1]  # Default to last timestamp
        start_time = segment["timestamp"].iloc[0]
        delta_m = segment["kiln_weight_avg"].iloc[-1] - segment["kiln_weight_avg"].iloc[0]
        total_delta_t = (segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0]).total_seconds()
        expected_slope = delta_m / total_delta_t  # kg/second

        # Look at the last 30% of the segment to find where emptying slows down
        start_idx = int(0.7 * len(segment))
        for i in range(start_idx, len(segment) - 1):
            # Calculate actual rate of change at this point
            actual_rate = (
                segment["kiln_weight_avg"].iloc[i + 1] - segment["kiln_weight_avg"].iloc[i]
            ) / 3  # kg/s (3-second intervals)

            # If the rate becomes much smaller than expected (less than 1% of expected slope magnitude)
            if abs(actual_rate) < 0.01 * abs(expected_slope):
                end_time = segment["timestamp"].iloc[i]
                break

        delta_t = (end_time - start_time).total_seconds()  # seconds

        if (
            segment["kiln_weight_avg"].mean() < 100
            or abs(delta_m) < 1000  # kg -- only consider segments with significant weight change
            or not (segment["kiln_weight_avg"].diff().dropna() < 50).all()
        ):
            continue

        average_RPM = np.array(segment["kiln_rpm"]).mean()  # RPM
        # Convert timestamp to seconds since start for polyfit
        # coefficients = np.polyfit(np.linspace(0, delta_t., len(segment)), segment["kiln_weight_avg"], 1)

        k = delta_m / (delta_t - average_RPM) * 3600  # kg/RPM*hour
        ks.append(k)  # slope -- kg/RPM*hour, basically mass flowing out due to RPM
        rpms.append(average_RPM)

        helper.append(segment)

        print(k)
        # plot_segmented_kiln_data(segment)  # Plot each segment with its slope

        # print(delta_t, delta_m, average_RPM, segment["kiln_weight_avg"].mean(), k)

    # plt.scatter(rpms, ks, color="r", linewidth=1, alpha=0.7)
    # plt.xlabel("RPM")
    # plt.ylabel("k (kg/RPM*seconds)")
    # plt.title("RPM vs k (kg/RPM*seconds)")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    # print(len(ks))
    # print(ks)
    return np.array(ks).mean()  # convert to kg/RPM*hour


def calculate_massflow(time_start, time_end, k, df):
    # Filter the dataframe to the time range
    mask = (df["timestamp"] >= time_start) & (df["timestamp"] <= time_end)
    segment_df = df[mask]

    if len(segment_df) < 2:
        return np.nan  # Not enough data points

    delta_m = segment_df["kiln_weight_avg"].iloc[-1] - segment_df["kiln_weight_avg"].iloc[0]
    delta_t = (segment_df["timestamp"].iloc[-1] - segment_df["timestamp"].iloc[0]).total_seconds() / 3600  # Hours
    average_RPM = segment_df["kiln_rpm"].mean()

    # k is negative, so we subtract it from the mass flow calculation
    mass_flow = delta_m / delta_t - k * average_RPM
    return mass_flow


def main():
    """
    Main function to load and plot kiln weight and RPM data
    """
    print("📊 ARMAN WEIGHT ANALYSIS")
    print("Usage: python arman_weight_analysis.py [--refresh]")

    # Example date range - modify these as needed
    start_date = "2025-02-01T00:00:00"  # February 1st 12am
    end_date = "2025-06-15T00:00:00"  # June 15th 12am

    print(f"\n🔍 Loading kiln data for analysis...")

    # Load the data
    df = load_kiln_data(start_date, end_date, "analysis")

    # Calculate k once before the loop
    print("Calculating k_RPM coefficient...")
    k = solve_k_RPM(df)
    print(f"Using k = {k}")
    # print(solve_k_RPM(df))
    # Plot all data to see what's happening
    # plot_kiln_data(df, f" ({start_date} to {end_date})")

    arman_df = pd.read_csv("scout_full_Xy.csv")
    copy_df = arman_df.copy()
    feature_vector_df = load_kiln_data(
        arman_df["window_start_utc"].min(),
        str(pd.to_datetime(arman_df["window_end_utc"].max()) + pd.Timedelta(minutes=40)),
        "data_for_scout",
    )

    for start_time, end_time in zip(arman_df["window_start_utc"], arman_df["window_end_utc"]):
        start_time = pd.to_datetime(start_time, utc=True).tz_convert(pacific_tz)
        end_time = pd.to_datetime(end_time, utc=True).tz_convert(pacific_tz)

        # print(f"Analyzing segment from {start_time} to {end_time}")

        # Calculate mass flow for this segment using the pre-calculated k
        mass_flow = calculate_massflow(start_time, end_time, k, feature_vector_df)

        # Use boolean indexing to update the dataframe
        mask = copy_df["window_start_utc"] == start_time.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
        copy_df.loc[mask, "mass_flow"] = mass_flow

    copy_df.to_csv("scout_full_Xy_with_mass_flow.csv", index=False)
    # for segment in solve_k_RPM(df):
    # Plot each segment to see what's happening
    # plot_segmented_kiln_data(segment)
    print("Percent of negative mass flow values:")
    neg = 0
    for i in range(len(copy_df)):
        if copy_df["mass_flow"][i] < 0:
            print(f"Negative mass flow at index {i}: {copy_df['mass_flow'][i]}")
            neg += 1
    print(f"Total negative mass flow values: {neg / copy_df.shape[0] * 100:.2f}%")


if __name__ == "__main__":
    main()
