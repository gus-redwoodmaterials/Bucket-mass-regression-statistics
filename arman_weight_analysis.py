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

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

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
    "bucket_mass": "rc1/4420-calciner/acmestatus/bucket_mass",
    "large_mod_pusher": "RC1/PLC9-Module_Loading/4420-FV-0021/Extend_Percentage".lower(),
    "c_large_mod": "rc1/4420-calciner/acmestatus/bucket_mass_regress_c_large_mod_mass",
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

    print(df.head())  # Print first few rows for verification
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
        # first bit is selecting segments with enough/ approrpiate data
        if (
            segment[["kiln_weight_avg", "kiln_rpm"]].isna().any().any()
            or len(segment) < 3600 / 2 / DATA_TIMESTEP_SECONDS
        ):
            continue

        end_time = segment["timestamp"].iloc[-1]  # Default to last timestamp
        start_time = segment["timestamp"].iloc[0]
        delta_m = segment["kiln_weight_avg"].iloc[-1] - segment["kiln_weight_avg"].iloc[0]
        for i in range(int(0.5 * len(segment) - 1), len(segment)):
            diff = segment["kiln_weight_avg"].iloc[i] - segment["kiln_weight_avg"].iloc[i - 1]
            if diff < 0.01 * delta_m:
                end_time = segment["timestamp"].iloc[i]
                break

        delta_t = (end_time - start_time).total_seconds()  # seconds

        if (
            segment["kiln_weight_avg"].mean() < 100
            or delta_m > -1000  # kg -- only consider segments with significant weight change
            or segment["kiln_weight_avg"].diff().max() > 100  # steep jumps up
            or segment["kiln_weight_avg"].diff().abs().max() > 500  # large discontinuities
        ):
            continue

        average_RPM = np.array(segment["kiln_rpm"]).mean()  # RPM
        if average_RPM < 0.1:
            continue
        # Convert timestamp to seconds since start for polyfit
        # coefficients = np.polyfit(np.linspace(0, delta_t., len(segment)), segment["kiln_weight_avg"], 1)

        # this section actually calculates the slope k
        k = delta_m / (delta_t * average_RPM) * 3600  # kg/RPM*hour
        if abs(k) < 1000 or abs(k) > 10000:  # filter out extreme values
            continue
        ks.append(k)  # slope -- kg/RPM*hour, basically mass flowing out due to RPM
        rpms.append(average_RPM)

        helper.append(segment)

        print(k)
        if abs(k) > 6000 or abs(k) < 2000:
            plot_segmented_kiln_data(segment)  # Plot each segment with its slope

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


def calculate_massflow_method2(time_start, time_end, k, df):
    # Filter the dataframe to the time range
    mask = (df["timestamp"] >= time_start) & (df["timestamp"] <= time_end)
    segment_df = df[mask]

    if len(segment_df) < 2:
        return np.nan  # Not enough data points

    delta_m = segment_df["kiln_weight_avg"].iloc[-1] - segment_df["kiln_weight_avg"].iloc[0]
    delta_t = (segment_df["timestamp"].iloc[-1] - segment_df["timestamp"].iloc[0]).total_seconds() / 3600  # Hours
    average_RPM = segment_df["kiln_rpm"].mean()

    print(delta_m, delta_t, average_RPM, k, len(segment_df))
    # k is negative, so we subtract it from the mass flow calculation
    mass_flow = delta_m / delta_t - k * average_RPM
    if mass_flow < 0:
        mass_flow = 0  # Ensure non-negative mass flow

    return mass_flow


# Small mod -- number of buckets counting code
# speed on conveyor in hz
HZ_CONVEY_CURVE = [0, 0.5, 1, 2, 4, 6, 9, 12, 15, 17, 20]
DEFAULT_BUCKET_MASS = 30  # kg

# seconds per bucket, (values less than 2 hz don't slow down the conveyor)
# so the sec_per_bucket has 20 seconds subtracted from it
# values above 2 hz, the conveyor is slowed down and need to add 20 sec for this
SEC_PER_BUCKET_CURVE = [1e9, 300, 140, 60, 29, 21, 13, 9, 7, 6, 5]


def number_buckets_from_coneyor_hz(hz_infeed_convey):
    """
    Use interp curve to convert infeed conveyor Hz to buckets/sec.
    """
    sec_per_bucket = np.interp(hz_infeed_convey, HZ_CONVEY_CURVE, SEC_PER_BUCKET_CURVE)

    return 1 / sec_per_bucket


def count_large_mods_in_window(df, start_time, end_time):
    """Count large modules loaded in time window"""
    mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
    pusher_data = df[mask]["large_mod_pusher"]

    if len(pusher_data) == 0:
        return 0

    # Convert to binary (50% threshold)
    pusher_binary = (pusher_data >= 50).astype(int)

    # Count transitions from 0 to 1 (closed to open)
    transitions = np.sum(np.diff(pusher_binary) == 1)

    return transitions


def calculate_mass_flow_method3(start_time, end_time, df):
    """
    Calculate mass flow using feedrate method based on small and large module counts.

    Returns mass flow in kg/hour
    """
    # Filter the dataframe to the time range
    mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
    segment_df = df[mask]

    if len(segment_df) < 2:
        return np.nan, np.nan, np.nan

    # Calculate time window in hours
    delta_t_hours = (end_time - start_time).total_seconds() / 3600

    # Get average bucket mass during this window
    average_bucket_mass = segment_df["bucket_mass"].mean()  # kg per bucket
    if pd.isna(average_bucket_mass):
        average_bucket_mass = DEFAULT_BUCKET_MASS

    small_mod_chunks = np.array_split(
        segment_df["small_mod_feedrate"], len(segment_df["small_mod_feedrate"]) * DATA_TIMESTEP_SECONDS / 60
    )  # should chunk the data into 1 minute segments
    tot_average_feed = 0
    for chunk in small_mod_chunks:
        if chunk.empty:
            continue
        # Calculate the maximum feedrate in this chunk
        max_feedrate = chunk.max()
        print(f"Max feedrate in chunk: {max_feedrate} Hz")
        tot_average_feed += max_feedrate

    tot_average_feed /= len(small_mod_chunks)  # Average over all chunks
    print(f"Total average feed in window: {tot_average_feed} Hz")
    # Count small modules (buckets) in this window
    small_mod_count = number_buckets_from_coneyor_hz(tot_average_feed) * 3600  # buckets/hour
    print(number_buckets_from_coneyor_hz(tot_average_feed) * 60, "buckets/minute")
    small_mod_massflow = small_mod_count * average_bucket_mass  # -- kg/hr
    # if small_mod_massflow > 10000:
    #     print(
    #         f"bucket_mass: {average_bucket_mass}, small_mod_count: {small_mod_count}, small_mod_massflow: {small_mod_massflow}"
    #     )

    # Count large modules and get their mass coefficient
    large_mod_count = count_large_mods_in_window(df, start_time, end_time)
    c_large_mod = segment_df["c_large_mod"].mean()  # kg per large module
    if pd.isna(c_large_mod):
        c_large_mod = 0  # Default if no large module data

    large_mod_massflow = large_mod_count * c_large_mod / delta_t_hours  # kg/hr

    mass_flow_rate = small_mod_massflow + large_mod_massflow  # Total mass flow rate (kg/hour)

    return small_mod_massflow, large_mod_massflow, mass_flow_rate


def main():
    """
    Main function to load and plot kiln weight and RPM data
    """
    print("📊 ARMAN WEIGHT ANALYSIS")
    print("Usage: python arman_weight_analysis.py [--refresh]")

    # Example date range - modify these as needed
    start_date = "2025-06-14T12:00:00"  # June 14th 12pm
    end_date = "2025-06-15T00:00:00"  # June 15th 12am

    print(f"\n🔍 Loading kiln data for analysis...")

    # Load the data
    df = load_kiln_data(start_date, end_date, "analysis")
    print(df.head())

    # Calculate k once before the loop
    print("Calculating k_RPM coefficient...")
    # k = solve_k_RPM(df)
    # print(f"Using k = {k}")
    # print(solve_k_RPM(df))
    # Plot all data to see what's happening
    # plot_kiln_data(df, f" ({start_date} to {end_date})")

    # arman_df = pd.read_csv("scout_full_Xy.csv")
    # copy_df = arman_df.copy()
    # feature_vector_df = load_kiln_data(
    #     arman_df["window_start_utc"].min(),
    #     str(pd.to_datetime(arman_df["window_end_utc"].max()) + pd.Timedelta(minutes=40)),
    #     "data_for_scout",
    # )
    # ks = [-2000, -6000]
    # for k in ks:
    #     for start_time, end_time in zip(arman_df["window_start_utc"], arman_df["window_end_utc"]):
    #         start_time = pd.to_datetime(start_time, utc=True).tz_convert(pacific_tz)
    #         end_time = pd.to_datetime(end_time, utc=True).tz_convert(pacific_tz)

    #         # print(f"Analyzing segment from {start_time} to {end_time}")

    #         # Calculate mass flow for this segment using the pre-calculated k
    #         flow = calculate_massflow_method2(start_time, end_time, k, feature_vector_df)

    #         # Use boolean indexing to update the dataframe
    #         mask = copy_df["window_start_utc"] == start_time.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    #         copy_df.loc[mask, f"mass_flow_{k}"] = flow

    # copy_df.to_csv("scout_full_Xy_massflow_2or6.csv", index=False)

    # print("Percent of negative mass flow values:")
    # neg = 0
    # for i in range(len(copy_df)):
    #     if copy_df["mass_flow"][i] < 0:
    #         neg += 1
    #         # copy_df["mass_flow"][i] = 0
    # print(f"Total negative mass flow values: {neg / copy_df.shape[0] * 100:.2f}%")


if __name__ == "__main__":
    main()
