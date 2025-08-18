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
    "bucket_mass": "rc1/4420-calciner/acmestatus/bucket_mass",
    "bucket_mass_regress": "rc1/4420-calciner/acmestatus/bucket_mass_regress_c_large_mod_mass",
    "bucket_mass_rpm_const": "rc1/4420-calciner/acmestatus/bucket_mass_regress_c_rpm_const",
    "bucket_mass_resid": "rc1/4420-calciner/acmestatus/bucket_mass_regress_resid",
    "infeed_hz_sp": "rc1/4420-calciner/acmestatus/hz_infeed_convey_mpc_ts_now",
    "large_module_cycle_time": "rc1/4420-calciner/acmestatus/module_cycle_time_mpc_ts_now",
    "cycle_time_complete": "rc1/4420-calciner/module_loading/hmi/cycle_time_complete/value",
    "infeed_hz_actual": "rc1/4420-calciner/4420-cvr-001/status/speed_feedback_hz",
    "kiln_weight_avg": "RC1/4420-Calciner/HMI/KilnWeightAvg/Value".lower(),
    "kiln_rpm": "rc1/4420-calciner/4420-kln-001_rpm/input",
    "kiln_weight_setpoint": "rc1/4420-calciner/hmi/4420-kln-001_weight_sp/value",
    "acme_alive": "rc1/4420-calciner/acme/acmefeedrate/acmealive",
    "acme_engineering": "rc1/4420-calciner/acme/acmefeedrate/acmeengineeringenable",
    "feed_robot_mode": "rc1/plc8-infeed_robot/hmi/feed_robot_mode/value",
    "main_fan_speed": "rc1/4430-exhaust/4430-fan-004/status/speed_feedback_hz",
    "startup_on": "rc1/4420-calciner/acme/acmefeedrate/kiln_bed_build_mode_honor_request",
    "tipper_one_on": "RC1/4410-FeedPrep/4410-VFR-001/Status/Speed_Feedback_Hz".lower(),
    "tipper_two_on": "RC1/4410-FeedPrep/4410-VFR-002/Status/Speed_Feedback_Hz".lower(),
}

# Database configuration
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")
DATA_TIMESTEP_SECONDS = 3
SMOOTH_SIDE_POINTS = 5  # ±5 → 11-point window


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

    # Filter for operational periods: acme_alive=True AND acme_engineering=True AND feed_robot_mode=False
    initial_rows = len(df)

    # Only look at times where acme is running without the robot
    df = df.loc[(df["acme_alive"] == 1) & (df["acme_engineering"] == 1) & (df["feed_robot_mode"] == 0)]
    # plt.plot(df["timestamp"], df["kiln_weight_avg"], label="Kiln Weight Avg")
    # plt.xlabel("Time")
    # plt.ylabel("Kiln Weight Avg (kg)")
    # plt.title("Kiln Weight Average Over Time")
    # plt.legend()
    # plt.show()

    filtered_rows = len(df)

    if filtered_rows > 0:
        print(
            f"   Filtered to {filtered_rows}/{initial_rows} operational rows ({100 * filtered_rows / initial_rows:.1f}%)"
        )
    else:
        print(f"   ⚠️  No operational data found (acme_alive=True, acme_engineering=True, feed_robot_mode=False)")

    print(f"   Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# Constants
DATA_TIMESTEP_SECONDS = 3  # 1 sample every 3 s
SMOOTH_SIDE_POINTS = 5  # ±5 → 11-point window


def analyze_kiln_weight_rate(df, rate_threshold_kg_per_hr=2000, rate_window_minutes=20):
    """
    Sliding‐window rate analysis with simple smoothing.

    Returns
    -------
    dict
        high_rate_events  : int     – number of contiguous excursions
        max_rate          : float   – peak kg/h
        max_rate_time     : pd.Timestamp | None
        rate_threshold    : float   – echo input
    """
    if {"kiln_weight_avg", "timestamp"} - set(df.columns):
        return {"high_rate_events": 0, "max_rate": 0, "max_rate_time": None, "rate_threshold": rate_threshold_kg_per_hr}

    # --- clean & sort ---
    controller_hours = len(df) * DATA_TIMESTEP_SECONDS / 3600  # convert seconds to hours
    cutoff = df["kiln_weight_setpoint"].min() * 0.5  # only consider high enough weights
    df = df[df["kiln_weight_avg"] > cutoff].dropna(subset=["kiln_weight_avg", "timestamp"])
    # Only exclude rows where startup_on == 1 (in startup); allow NaN and 0
    if "startup_on" in df.columns:
        df = df[(df["startup_on"].isna()) | (df["startup_on"] == 0)]
    clean = (
        df[["timestamp", "kiln_weight_avg"]]
        .dropna()
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None))
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    rate_window_pts = int(rate_window_minutes * 60 / DATA_TIMESTEP_SECONDS)
    if len(clean) <= rate_window_pts:
        return {"high_rate_events": 0, "max_rate": 0, "max_rate_time": None, "rate_threshold": rate_threshold_kg_per_hr}

    rates = []
    rate_time = []
    i = 0
    while i < len(clean) - rate_window_pts:
        time_d_seconds = (clean["timestamp"].iloc[i + rate_window_pts] - clean["timestamp"].iloc[i]).total_seconds()
        time_d_minutes = time_d_seconds / 60
        time_d_hours = time_d_seconds / 3600
        if time_d_minutes > rate_window_minutes:
            i += 1
            continue
        delta_m = clean["kiln_weight_avg"].iloc[i + rate_window_pts] - clean["kiln_weight_avg"].iloc[i]

        rate = delta_m / (time_d_hours)  # kg/h
        if rate >= rate_threshold_kg_per_hr:
            rates.append(rate)
            rate_time.append(clean["timestamp"].iloc[i + rate_window_pts])
            i += rate_window_pts  # skip ahead to next window
        else:
            i += 1
    rates = np.array(rates)
    rate_time = np.array(rate_time)

    high_rate_events = len(rates)
    if high_rate_events == 0:
        return {"high_rate_events": 0, "max_rate": 0, "max_rate_time": None, "rate_threshold": rate_threshold_kg_per_hr}

    # --- 4) stats for the excursion with the absolute peak ---
    max_idx = np.argmax(rates)
    max_rate = rates[max_idx]
    peak_time = rate_time[max_idx]

    return {
        "high_rate_events": high_rate_events,
        "max_rate": float(max_rate),
        "max_rate_time": peak_time,
        "rate_threshold": rate_threshold_kg_per_hr,
        "rate_spikes_per_hour": high_rate_events / controller_hours if controller_hours > 0 else 0,
    }


def analyze_main_fan_spikes(
    df, fan_spike_threshold_hz=53.5, rolling_window_seconds=40, feeding_rate_threshold_kg_per_hr=1500
):
    """
    Analyze main fan spikes using Acme's algorithm and correlate with feeding rate increases

    Args:
        df: DataFrame with main_fan_speed, kiln_weight_avg, and timestamp columns
        fan_spike_threshold_hz: Fan speed threshold for spike detection (default: 53.5 Hz)
        rolling_window_seconds: Rolling average window for fan spike detection (default: 40 seconds)
        feeding_rate_threshold_kg_per_hr: Threshold for "sharp" feeding increases (default: 1500 kg/hr)

    Returns:
        Dictionary with fan spike analysis results
    """
    if "main_fan_speed" not in df.columns or "timestamp" not in df.columns:
        return None

    # Create a copy and remove NaN values
    df_clean = df[["timestamp", "main_fan_speed", "kiln_weight_avg"]].dropna().copy()

    if len(df_clean) < 200:  # Need sufficient data for rolling average
        return {"fan_spikes": 0, "spike_events": [], "correlation_analysis": None}

    # Sort by timestamp
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

    # Calculate 40-second rolling average of fan speed
    # For 3-second data, 40 seconds = ~13-14 data points
    rolling_window_points = max(5, int(rolling_window_seconds / DATA_TIMESTEP_SECONDS))

    df_clean["fan_speed_rolling_avg"] = (
        df_clean["main_fan_speed"]
        .rolling(window=rolling_window_points, center=True, min_periods=max(3, rolling_window_points // 2))
        .mean()
    )

    # Forward/backward fill any remaining NaN values at edges
    df_clean["fan_speed_rolling_avg"] = df_clean["fan_speed_rolling_avg"].bfill().ffill()

    # Identify fan spikes using Acme's algorithm
    fan_spike_mask = df_clean["fan_speed_rolling_avg"] > fan_spike_threshold_hz

    # Find fan spike events (consecutive spikes are grouped as one event)
    fan_spike_events = []
    in_spike = False
    spike_start = None

    i = 0
    while i < len(fan_spike_mask):
        is_spike = fan_spike_mask.iloc[i] if hasattr(fan_spike_mask, "iloc") else fan_spike_mask[i]
        if is_spike and not in_spike:
            # Start of a new spike event
            in_spike = True
            spike_start = i
            i += 1
        elif not is_spike and in_spike:
            # End of spike event
            in_spike = False
            spike_end = i - 1

            # Record the spike event
            spike_duration_seconds = (
                df_clean.iloc[spike_end]["timestamp"] - df_clean.iloc[spike_start]["timestamp"]
            ).total_seconds()
            max_fan_speed_idx = df_clean.iloc[spike_start : spike_end + 1]["fan_speed_rolling_avg"].idxmax()

            fan_spike_events.append(
                {
                    "start_time": df_clean.iloc[spike_start]["timestamp"],
                    "end_time": df_clean.iloc[spike_end]["timestamp"],
                    "duration_seconds": spike_duration_seconds,
                    "max_fan_speed": df_clean.iloc[max_fan_speed_idx]["fan_speed_rolling_avg"],
                    "max_fan_time": df_clean.iloc[max_fan_speed_idx]["timestamp"],
                    "start_idx": spike_start,
                    "end_idx": spike_end,
                }
            )

            # Jump forward 20 minutes after the end of the spike
            jump_minutes = 15
            end_time = df_clean.iloc[spike_end]["timestamp"]
            found_jump = False
            for j in range(spike_end + 1, len(df_clean)):
                if (df_clean.iloc[j]["timestamp"] - end_time).total_seconds() >= jump_minutes * 60:
                    i = j
                    found_jump = True
                    break
            if not found_jump:
                break  # No more data at least 20 min after spike
        else:
            i += 1

    # Handle case where spike extends to end of data
    if in_spike:
        spike_end = len(df_clean) - 1
        spike_duration_seconds = (
            df_clean.iloc[spike_end]["timestamp"] - df_clean.iloc[spike_start]["timestamp"]
        ).total_seconds()
        max_fan_speed_idx = df_clean.iloc[spike_start : spike_end + 1]["fan_speed_rolling_avg"].idxmax()

        fan_spike_events.append(
            {
                "start_time": df_clean.iloc[spike_start]["timestamp"],
                "end_time": df_clean.iloc[spike_end]["timestamp"],
                "duration_seconds": spike_duration_seconds,
                "max_fan_speed": df_clean.iloc[max_fan_speed_idx]["fan_speed_rolling_avg"],
                "max_fan_time": df_clean.iloc[max_fan_speed_idx]["timestamp"],
                "start_idx": spike_start,
                "end_idx": spike_end,
            }
        )

    total_time_in_df = df_clean.shape[0] * DATA_TIMESTEP_SECONDS / 3600  # Total time in hours
    # Calculate time span and daily average
    total_time_hours = (df_clean["timestamp"].iloc[-1] - df_clean["timestamp"].iloc[0]).total_seconds() / 3600
    total_time_days = total_time_hours / 24
    spikes_per_day = len(fan_spike_events) / total_time_days if total_time_days > 0 else 0
    fan_spike_per_hour_acme = len(fan_spike_events) / total_time_in_df if total_time_in_df > 0 else 0

    print(f"   Fan spike analysis ({rolling_window_seconds}s rolling avg > {fan_spike_threshold_hz} Hz):")
    print(f"   Total fan spikes: {len(fan_spike_events)} over {total_time_days:.1f} days")
    print(f"   Average spikes per day: {spikes_per_day:.1f}")
    print(f"   Fan spikes per hour (controller running): {fan_spike_per_hour_acme:.3f}")

    return {
        "fan_spikes": len(fan_spike_events),
        "spike_events": fan_spike_events,
        "spikes_per_day": spikes_per_day,
        "fan_spike_per_hour_acme": fan_spike_per_hour_acme,
        "total_time_days": total_time_days,
        "fan_spike_threshold_hz": fan_spike_threshold_hz,
        "rolling_window_seconds": rolling_window_seconds,
        "df_clean": df_clean,  # Include for plotting if needed
    }


def plot_kiln_weight_rate_analysis(rate_analysis_results, highlight_threshold=5000, start_time=None, end_time=None):
    """
    Plot kiln weight and rate analysis with highlighted high-rate sections

    Args:
        rate_analysis_results: Dictionary from analyze_kiln_weight_rate()
        highlight_threshold: Rate threshold for highlighting (default: 5000 kg/hr)
        start_time: Optional start time for filtering (e.g., "2025-05-05 09:00:00")
        end_time: Optional end time for filtering (e.g., "2025-05-05 10:00:00")
    """
    if not rate_analysis_results or "df_clean" not in rate_analysis_results:
        print("   ❌ No rate analysis data available for plotting")
        return

    df_clean = rate_analysis_results["df_clean"]
    valid_rates = rate_analysis_results["valid_rates"]

    # Filter data by time range if specified
    if start_time and end_time:
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)

        # Filter df_clean
        time_mask_clean = (df_clean["timestamp"] >= start_dt) & (df_clean["timestamp"] <= end_dt)
        df_clean_filtered = df_clean[time_mask_clean].copy()

        # Filter valid_rates
        time_mask_rates = (valid_rates["timestamp"] >= start_dt) & (valid_rates["timestamp"] <= end_dt)
        valid_rates_filtered = valid_rates[time_mask_rates].copy()

        print(f"   📊 Plotting filtered time range: {start_time} to {end_time}")
        print(f"   Filtered to {len(df_clean_filtered)} weight points and {len(valid_rates_filtered)} rate points")
    else:
        df_clean_filtered = df_clean
        valid_rates_filtered = valid_rates

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Kiln weight (raw and smoothed)
    ax1.plot(
        df_clean_filtered["timestamp"],
        df_clean_filtered["kiln_weight_avg"],
        color="lightblue",
        alpha=0.6,
        linewidth=1,
        label="Raw Kiln Weight",
    )
    ax1.plot(
        df_clean_filtered["timestamp"],
        df_clean_filtered["weight_smoothed"],
        color="blue",
        linewidth=2,
        label="Smoothed Kiln Weight",
    )

    # Highlight sections where rate exceeds threshold
    high_rate_mask = valid_rates_filtered["rate_kg_per_hr"] > highlight_threshold
    if high_rate_mask.any():
        high_rate_times = valid_rates_filtered.loc[high_rate_mask, "timestamp"]
        # For the weight plot, we need to find corresponding smoothed weights
        high_rate_weights = []
        for timestamp in high_rate_times:
            # Find closest timestamp in df_clean_filtered
            closest_idx = (df_clean_filtered["timestamp"] - timestamp).abs().idxmin()
            high_rate_weights.append(df_clean_filtered.loc[closest_idx, "weight_smoothed"])

        ax1.scatter(
            high_rate_times,
            high_rate_weights,
            color="red",
            s=50,
            zorder=5,
            alpha=0.8,
            label=f"Rate >{highlight_threshold} kg/hr",
        )

        print(f"   📍 Highlighted {len(high_rate_times)} time periods with rate >{highlight_threshold} kg/hr")

    ax1.set_ylabel("Kiln Weight (kg)")
    title_suffix = f" - {start_time} to {end_time}" if start_time and end_time else ""
    ax1.set_title(
        f"Kiln Weight Analysis (Smoothing: {rate_analysis_results['smoothing_window_minutes']} min, Rate Window: {rate_analysis_results['rate_window_minutes']} min, Segments: {rate_analysis_results.get('continuous_segments', 'N/A')}){title_suffix}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rate of change over time windows
    ax2.plot(
        valid_rates_filtered["timestamp"],
        valid_rates_filtered["rate_kg_per_hr"],
        color="green",
        linewidth=1,
        alpha=0.7,
        label=f"Rate of Change ({rate_analysis_results['rate_window_minutes']}-min windows)",
    )

    # Highlight high rates
    if high_rate_mask.any():
        ax2.scatter(
            valid_rates_filtered.loc[high_rate_mask, "timestamp"],
            valid_rates_filtered.loc[high_rate_mask, "rate_kg_per_hr"],
            color="red",
            s=50,
            zorder=5,
            alpha=0.8,
            label=f"Rate >{highlight_threshold} kg/hr",
        )

    # Add threshold lines
    ax2.axhline(
        y=highlight_threshold, color="red", linestyle="--", alpha=0.7, label=f"{highlight_threshold} kg/hr threshold"
    )
    ax2.axhline(y=1500, color="orange", linestyle="--", alpha=0.7, label="1500 kg/hr (1 ton/hr)")

    ax2.set_ylabel("Rate (kg/hr)")
    ax2.set_xlabel("Time")
    ax2.set_title(
        f"Rate of Kiln Weight Change (calculated over {rate_analysis_results['rate_window_minutes']}-minute windows){title_suffix}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Print summary of highlighted events
    if high_rate_mask.any():
        print(f"\n   📋 HIGH RATE EVENTS SUMMARY:")
        high_rate_events = valid_rates_filtered[high_rate_mask].sort_values("rate_kg_per_hr", ascending=False)
        for i, (idx, row) in enumerate(high_rate_events.head(5).iterrows()):
            print(f"   {i + 1}. {row['timestamp']}: {row['rate_kg_per_hr']:.0f} kg/hr")
            print(f"      Weight change: {row['weight_change']:.1f} kg over {row['time_window_hours']:.2f} hours")
            print(f"      From {row['weight_start']:.1f} to {row['weight_end']:.1f} kg")


def analysis(df, description="dataset"):
    """
    Perform analysis on the loaded DataFrame

    Args:
        df: DataFrame containing the feedrate control data
        description: Description of the dataset (e.g., "before_controller", "after_controller")
    """
    print(f"\n🔍 Analyzing {description}...")

    overshoot_results = overshoot_analysis(df)
    rate_analysis = analyze_kiln_weight_rate(df)
    fan_analysis = analyze_main_fan_spikes(df)

    # Print overshoot results
    if overshoot_results:
        print(f"\n   📊 KILN WEIGHT OVERSHOOT ANALYSIS:")
        print(f"   Total data points: {overshoot_results['total_points']}")
        print(
            f"   Overshoot events: {overshoot_results['overshoot_events']} ({overshoot_results['overshoot_percentage']:.1f}% of time)"
        )
        print(f"   Average overshoot magnitude: {overshoot_results['avg_overshoot_magnitude']:.2f} kg")
        print(
            f"   Maximum overshoot: {overshoot_results['max_overshoot']:.2f} kg at {overshoot_results['max_overshoot_time']}"
        )
        print(f"   Overshoot std deviation: {overshoot_results['std_overshoot']:.2f} kg")
        if "avg_setpoint" in overshoot_results and overshoot_results["avg_setpoint"] is not None:
            print(f"   Average setpoint: {overshoot_results['avg_setpoint']:.2f} kg")
            print(f"   Relative overshoot: {overshoot_results['relative_overshoot_pct']:.2f}% of setpoint")
        print(f"   Total time considered: {overshoot_results['total_time_hrs']:.2f} hours")
        print(f"   Percent of time tippers were on: {overshoot_results['tipper_percentage']:.2f}%")
    else:
        print(f"\n   📊 KILN WEIGHT OVERSHOOT ANALYSIS:")
        print(f"   No overshoot analysis available.")

    # Print rate analysis
    if rate_analysis:
        print(f"\n   📈 KILN WEIGHT RATE ANALYSIS:")
        print(f"   Rate Spikes: {rate_analysis['high_rate_events']}")
        print(f"   Maximum rate increase: {rate_analysis['max_rate']:.1f} kg/hr")
        if rate_analysis["max_rate_time"]:
            print(f"   Maximum rate occurred at: {rate_analysis['max_rate_time']}")
        print(f"   Rate spikes per hour (controller running): {rate_analysis['rate_spikes_per_hour']:.3f}")

    # Print fan analysis
    if fan_analysis:
        print(f"\n   🌪️  MAIN FAN SPIKE ANALYSIS:")
        print(f"   Total fan spikes: {fan_analysis['fan_spikes']}")
        print(f"   Average spikes per day: {fan_analysis['spikes_per_day']:.1f}")
        if "fan_spike_per_hour_acme" in fan_analysis:
            print(f"   Fan spikes per hour (controller running): {fan_analysis['fan_spike_per_hour_acme']:.3f}")

    # Return key metrics for comparison
    return {
        "description": description,
        "total_points": overshoot_results["total_points"] if overshoot_results else 0,
        "overshoot_events": overshoot_results["overshoot_events"] if overshoot_results else 0,
        "overshoot_percentage": overshoot_results["overshoot_percentage"] if overshoot_results else 0,
        "avg_overshoot_magnitude": overshoot_results["avg_overshoot_magnitude"] if overshoot_results else 0,
        "max_overshoot": overshoot_results["max_overshoot"] if overshoot_results else 0,
        "max_overshoot_time": overshoot_results["max_overshoot_time"] if overshoot_results else None,
        "std_overshoot": overshoot_results["std_overshoot"] if overshoot_results else 0,
        "avg_setpoint": overshoot_results["avg_setpoint"]
        if overshoot_results and "avg_setpoint" in overshoot_results
        else None,
        "relative_overshoot_pct": overshoot_results["relative_overshoot_pct"]
        if overshoot_results and "relative_overshoot_pct" in overshoot_results
        else None,
        "high_rate_events": rate_analysis["high_rate_events"] if rate_analysis else 0,
        "max_rate": rate_analysis["max_rate"] if rate_analysis else 0,
        "fan_spikes": fan_analysis["fan_spikes"] if fan_analysis else 0,
        "spikes_per_day": fan_analysis["spikes_per_day"] if fan_analysis else 0,
        "fan_spike_per_hour_acme": fan_analysis["fan_spike_per_hour_acme"]
        if fan_analysis and "fan_spike_per_hour_acme" in fan_analysis
        else 0,
    }


def overshoot_analysis(df):
    """
    Perform overshoot analysis on the loaded DataFrame
    Returns a dictionary of overshoot metrics
    """
    if "kiln_weight_avg" not in df.columns or "kiln_weight_setpoint" not in df.columns:
        print("   ❌ Missing kiln weight columns - cannot perform overshoot analysis")
        return None
    cutoff = df["kiln_weight_setpoint"].min() * 0.5
    overshoot = df[df["kiln_weight_avg"] > cutoff]
    tipper_helper = df[(df["tipper_one_on"] > 0.1) & (df["tipper_two_on"] > 0.1)]
    overshoot = df["kiln_weight_avg"] - df["kiln_weight_setpoint"]
    positive_overshoot = overshoot[overshoot > 0]
    total_points = len(df)
    overshoot_points = len(positive_overshoot)
    overshoot_percentage = (overshoot_points / total_points) * 100 if total_points > 0 else 0
    tipper_percentage = (
        (len(tipper_helper) / total_points) * 100 if total_points > 0 else 0
    )  # Percentage of time tipper is on
    total_time_hrs = total_points * DATA_TIMESTEP_SECONDS / 3600  # Total time in hours
    if len(positive_overshoot) > 0:
        avg_overshoot = positive_overshoot.mean()
        max_overshoot = positive_overshoot.max()
        std_overshoot = positive_overshoot.std()
        max_overshoot_idx = overshoot.idxmax()
        max_overshoot_time = df.loc[max_overshoot_idx, "timestamp"]
        avg_setpoint = df["kiln_weight_setpoint"].mean() if "kiln_weight_setpoint" in df.columns else None
        relative_overshoot = (avg_overshoot / avg_setpoint) * 100 if avg_setpoint and avg_setpoint > 0 else 0
        return {
            "total_points": total_points,
            "overshoot_events": overshoot_points,
            "overshoot_percentage": overshoot_percentage,
            "avg_overshoot_magnitude": avg_overshoot,
            "max_overshoot": max_overshoot,
            "max_overshoot_time": max_overshoot_time,
            "std_overshoot": std_overshoot,
            "avg_setpoint": avg_setpoint,
            "relative_overshoot_pct": relative_overshoot,
            "tipper_percentage": tipper_percentage,
            "total_time_hrs": total_time_hrs,
        }
    else:
        avg_setpoint = df["kiln_weight_setpoint"].mean() if "kiln_weight_setpoint" in df.columns else None
        return {
            "total_points": total_points,
            "overshoot_events": 0,
            "overshoot_percentage": 0,
            "avg_overshoot_magnitude": 0,
            "max_overshoot": 0,
            "max_overshoot_time": None,
            "std_overshoot": 0,
            "avg_setpoint": avg_setpoint,
            "relative_overshoot_pct": 0,
            "tipper_percentage": 0,
            "total_time_hrs": 0,
        }


def main():
    """
    Main function to load and compare feedrate control data
    """
    print("🔄 FEEDRATE CONTROLLER COMPARISON ANALYSIS")
    print("Usage: python feedrate_controller_comparison.py [--refresh] [--analyze]")

    # Check for analysis flag
    analyze_data = "--analyze" in sys.argv or "-a" in sys.argv

    # Example date ranges - modify these as needed
    none_start = "2025-08-08T21:30:00"  # August 8th 9:30pm
    none_end = "2025-08-13T06:24:00"  # August 13th 6:24am
    bucket_start = "2025-08-02T15:00:00"  # Resumed filling August 13th 4:20pm
    bucket_end = "2025-08-08T16:00:00"  # Modify this date to just be the most up to date
    derivative_start = "2025-08-14T12:41:00"  # Resumed filling August 14th 8:00am
    derivative_end = "2025-08-18T08:10:00"

    # Load June 4th data
    print(f"\n🔍 Loading data...")
    df_none = load_data(none_start, none_end, "no_controller")
    df_bucket = load_data(bucket_start, bucket_end, "bucket_controller")
    df_derivative = load_data(derivative_start, derivative_end, "derivative_controller")

    # Perform analysis if requested
    if True:
        print(f"\n{'=' * 60}")
        print("no_controller ANALYSIS")
        print(f"{'=' * 60}")
        results_none = analysis(df_none, "no controller")
        print(f"\n{'=' * 60}")
        print("bucket_controller ANALYSIS")
        print(f"{'=' * 60}")
        results_bucket = analysis(df_bucket, "bucket controller")
        print(f"\n{'=' * 60}")
        print("derivative_controller ANALYSIS")
        print(f"{'=' * 60}")
        results_derivative = analysis(df_derivative, "derivative controller")


if __name__ == "__main__":
    main()
