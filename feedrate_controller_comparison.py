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
    "kiln_weight_avg": "RC1/4420-Calciner/4420-WT-0007_Alarm/Input".lower(),
    "kiln_rpm": "rc1/4420-calciner/4420-kln-001_rpm/input",
    "kiln_weight_setpoint": "rc1/4420-calciner/hmi/4420-kln-001_weight_sp/value",
    "acme_alive": "rc1/4420-calciner/acme/acmefeedrate/acmealive",
    "acme_engineering": "rc1/4420-calciner/acme/acmefeedrate/acmeengineeringenable",
    "feed_robot_mode": "rc1/plc8-infeed_robot/hmi/feed_robot_mode/value",
    "main_fan_speed": "rc1/4430-exhaust/4430-fan-004/status/speed_feedback_hz",
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

    # Parse dates and localize to Pacific timezone
    start = pacific_tz.localize(parse(start_date_str))
    end = pacific_tz.localize(parse(end_date_str))

    # Generate filename based on date range and description
    start_str = start.strftime("%Y%m%d_%H%M")
    end_str = end.strftime("%Y%m%d_%H%M")
    csv_filename = f"feedrate_analysis_{description}_{start_str}_to_{end_str}.csv"
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

    # Filter out negative Hz values
    if "infeed_hz_actual" in df.columns:
        negative_hz_count = (df["infeed_hz_actual"] < 0).sum()
        if negative_hz_count > 0:
            print(f"   Cleaned {negative_hz_count} negative Hz values")
            df.loc[df["infeed_hz_actual"] < 0, "infeed_hz_actual"] = np.nan

    # Filter out negative kiln weight values (faulty data)
    if "kiln_weight_avg" in df.columns:
        negative_weight_count = (df["kiln_weight_avg"] < 0).sum()
        if negative_weight_count > 0:
            print(f"   Cleaned {negative_weight_count} negative kiln weight values")
            df.loc[df["kiln_weight_avg"] < 0, "kiln_weight_avg"] = np.nan

    # Filter for operational periods: acme_alive=True AND acme_engineering=True AND feed_robot_mode=False
    initial_rows = len(df)

    # Apply the operational filter
    df = df.loc[(df["acme_alive"] == 1) & (df["acme_engineering"] == 1) & (df["feed_robot_mode"] == 0)]
    plt.plot(df["timestamp"], df["kiln_weight_avg"], label="Kiln Weight Avg")
    plt.xlabel("Time")
    plt.ylabel("Kiln Weight Avg (kg)")
    plt.title("Kiln Weight Average Over Time")
    plt.legend()
    plt.show()

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


def analyze_kiln_weight_rate(df, rate_threshold_kg_per_hr=1500, rate_window_minutes=15):
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
    clean = (
        df[["timestamp", "kiln_weight_avg"]]
        .dropna()
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None))
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if clean.empty:
        return {"high_rate_events": 0, "max_rate": 0, "max_rate_time": None, "rate_threshold": rate_threshold_kg_per_hr}

    # --- 1) ±5-point moving mean ---
    smooth = (
        clean["kiln_weight_avg"]
        .rolling(window=2 * SMOOTH_SIDE_POINTS + 1, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    # --- 2) change in mass over 15 min (fixed lag) ---
    rate_window_pts = int(rate_window_minutes * 60 / DATA_TIMESTEP_SECONDS)  # ≈300
    if len(smooth) <= rate_window_pts:
        return {"high_rate_events": 0, "max_rate": 0, "max_rate_time": None, "rate_threshold": rate_threshold_kg_per_hr}

    delta_m = smooth[rate_window_pts:] - smooth[:-rate_window_pts]
    rates = delta_m * 3600 / (rate_window_minutes * 60)  # kg / h   (== delta_m * 4)
    rate_time = clean["timestamp"].iloc[rate_window_pts:].to_numpy()

    # --- 3) detect excursions above threshold ---
    above = rates > rate_threshold_kg_per_hr
    edges = np.diff(np.concatenate(([0], above.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]  # rising edge indices
    ends = np.where(edges == -1)[0] - 1  # falling edges

    high_rate_events = len(starts)
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
        feeding_rate_threshold_kg_per_hr: Threshold for "sharp" feeding increases (default: 1000 kg/hr)

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
    rolling_window_points = max(5, int(rolling_window_seconds / 3))

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

    for i, is_spike in enumerate(fan_spike_mask):
        if is_spike and not in_spike:
            # Start of a new spike event
            in_spike = True
            spike_start = i
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

    # Check if required columns exist
    if "kiln_weight_avg" not in df.columns or "kiln_weight_setpoint" not in df.columns:
        print("   ❌ Missing kiln weight columns - cannot perform overshoot analysis")
        return

    # Calculate overshoot: when actual weight exceeds setpoint
    overshoot = df["kiln_weight_avg"] - df["kiln_weight_setpoint"]

    # Only consider positive overshoots (when weight exceeds setpoint)
    positive_overshoot = overshoot[overshoot > 0]

    # Count overshoot events
    total_points = len(df)
    overshoot_points = len(positive_overshoot)
    overshoot_percentage = (overshoot_points / total_points) * 100 if total_points > 0 else 0

    if len(positive_overshoot) > 0:
        # Calculate overshoot statistics
        avg_overshoot = positive_overshoot.mean()
        max_overshoot = positive_overshoot.max()
        std_overshoot = positive_overshoot.std()

        # Find timestamp of maximum overshoot
        max_overshoot_idx = overshoot.idxmax()
        max_overshoot_time = df.loc[max_overshoot_idx, "timestamp"]

        print(f"\n   📊 KILN WEIGHT OVERSHOOT ANALYSIS:")
        print(f"   Total data points: {total_points}")
        print(f"   Overshoot events: {overshoot_points} ({overshoot_percentage:.1f}% of time)")
        print(f"   Average overshoot magnitude: {avg_overshoot:.2f} kg")
        print(f"   Maximum overshoot: {max_overshoot:.2f} kg at {max_overshoot_time}")
        print(f"   Overshoot std deviation: {std_overshoot:.2f} kg")

        # Additional statistics
        if "kiln_weight_setpoint" in df.columns:
            avg_setpoint = df["kiln_weight_setpoint"].mean()
            relative_overshoot = (avg_overshoot / avg_setpoint) * 100 if avg_setpoint > 0 else 0
            print(f"   Average setpoint: {avg_setpoint:.2f} kg")
            print(f"   Relative overshoot: {relative_overshoot:.2f}% of setpoint")

        # Analyze rate of change in kiln weight (smoothed)
        rate_analysis = analyze_kiln_weight_rate(df)
        if rate_analysis:
            print(f"\n   📈 KILN WEIGHT RATE ANALYSIS:")
            print(f"   Events with >1 ton/hr increase: {rate_analysis['high_rate_events']}")
            print(f"   Maximum rate increase: {rate_analysis['max_rate']:.1f} kg/hr")
            if rate_analysis["max_rate_time"]:
                print(f"   Maximum rate occurred at: {rate_analysis['max_rate_time']}")

        # Analyze main fan spikes
        fan_analysis = analyze_main_fan_spikes(df)
        if fan_analysis:
            print(f"\n   🌪️  MAIN FAN SPIKE ANALYSIS:")
            print(f"   Total fan spikes: {fan_analysis['fan_spikes']}")
            print(f"   Average spikes per day: {fan_analysis['spikes_per_day']:.1f}")
            if "fan_spike_per_hour_acme" in fan_analysis:
                print(f"   Fan spikes per hour (controller running): {fan_analysis['fan_spike_per_hour_acme']:.3f}")

        # Return key metrics for comparison
        return {
            "description": description,
            "total_points": total_points,
            "overshoot_events": overshoot_points,
            "overshoot_percentage": overshoot_percentage,
            "avg_overshoot_magnitude": avg_overshoot,
            "max_overshoot": max_overshoot,
            "max_overshoot_time": max_overshoot_time,
            "std_overshoot": std_overshoot,
            "avg_setpoint": avg_setpoint if "kiln_weight_setpoint" in df.columns else None,
            "relative_overshoot_pct": relative_overshoot if "kiln_weight_setpoint" in df.columns else None,
            "high_rate_events": rate_analysis["high_rate_events"] if rate_analysis else 0,
            "max_rate": rate_analysis["max_rate"] if rate_analysis else 0,
            "fan_spikes": fan_analysis["fan_spikes"] if fan_analysis else 0,
            "spikes_per_day": fan_analysis["spikes_per_day"] if fan_analysis else 0,
            "fan_spike_per_hour_acme": fan_analysis["fan_spike_per_hour_acme"]
            if fan_analysis and "fan_spike_per_hour_acme" in fan_analysis
            else 0,
        }
    else:
        print(f"\n   📊 KILN WEIGHT OVERSHOOT ANALYSIS:")
        print(f"   Total data points: {total_points}")
        print(f"   ✅ No overshoot events detected - weight stayed below setpoint")

        return {
            "description": description,
            "total_points": total_points,
            "overshoot_events": 0,
            "overshoot_percentage": 0,
            "avg_overshoot_magnitude": 0,
            "max_overshoot": 0,
            "std_overshoot": 0,
            "avg_setpoint": df["kiln_weight_setpoint"].mean() if "kiln_weight_setpoint" in df.columns else None,
            "relative_overshoot_pct": 0,
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
    before_start = "2025-06-04T10:00:00"  # June 4th 10am
    before_end = "2025-07-22T00:00:00"  # June 4th 11am
    after_start = "2025-07-22T00:00:00"  # Modify this date
    after_end = "2025-08-04T09:00:00"  # Modify this date

    # Load June 4th data
    print(f"\n🔍 Loading data...")
    df_before = load_data(before_start, before_end, "before_controller")
    df_after = load_data(after_start, after_end, "after_controller")

    # Perform analysis if requested
    if True:
        print(f"\n{'=' * 60}")
        print("before_controller ANALYSIS")
        print(f"{'=' * 60}")

        # Analyze May 8-9 data
        results_before = analysis(df_before, "before_controller")

        # Uncomment these lines when you have after data:
        results_after = analysis(df_after, "AFTER controller")

        # Compare results
        if results_before and results_after:
            print(f"\n📊 COMPARISON SUMMARY:")
            print(f"   BEFORE: {results_before['avg_overshoot_magnitude']:.2f} kg avg overshoot")
            print(f"   AFTER:  {results_after['avg_overshoot_magnitude']:.2f} kg avg overshoot")
            improvement = results_before["avg_overshoot_magnitude"] - results_after["avg_overshoot_magnitude"]
            print(
                f"   IMPROVEMENT: {improvement:.2f} kg ({improvement / results_before['avg_overshoot_magnitude'] * 100:.1f}% reduction)"
            )

            # Sharp increases in fan rate per day
            if (
                "high_rate_events" in results_before
                and "high_rate_events" in results_after
                and "total_points" in results_before
                and "total_points" in results_after
            ):
                # Calculate days for before and after
                before_days = results_before["total_points"] * 3 / 3600 / 24  # 3 sec per point
                after_days = results_after["total_points"] * 3 / 3600 / 24
                before_sharp_per_day = results_before["high_rate_events"] / before_days if before_days > 0 else 0
                after_sharp_per_day = results_after["high_rate_events"] / after_days if after_days > 0 else 0
                sharp_improvement = before_sharp_per_day - after_sharp_per_day
                print(f"   BEFORE: {before_sharp_per_day:.2f} sharp increases per day")
                print(f"   AFTER:  {after_sharp_per_day:.2f} sharp increases per day")
                print(
                    f"   IMPROVEMENT: {sharp_improvement:.2f} per day ({(sharp_improvement / before_sharp_per_day * 100) if before_sharp_per_day else 0:.1f}% reduction)"
                )

            # Fan spikes per day
            if "spikes_per_day" in results_before and "spikes_per_day" in results_after:
                before_spikes_per_day = results_before["spikes_per_day"]
                after_spikes_per_day = results_after["spikes_per_day"]
                spike_improvement = before_spikes_per_day - after_spikes_per_day
                print(f"   BEFORE: {before_spikes_per_day:.2f} fan spikes per day")
                print(f"   AFTER:  {after_spikes_per_day:.2f} fan spikes per day")
                print(
                    f"   IMPROVEMENT: {spike_improvement:.2f} per day ({(spike_improvement / before_spikes_per_day * 100) if before_spikes_per_day else 0:.1f}% reduction)"
                )

            # Fan spikes per hour (controller running)
            if "fan_spike_per_hour_acme" in results_before and "fan_spike_per_hour_acme" in results_after:
                before_spikes_per_hour = results_before["fan_spike_per_hour_acme"]
                after_spikes_per_hour = results_after["fan_spike_per_hour_acme"]
                spike_hour_improvement = before_spikes_per_hour - after_spikes_per_hour
                print(f"   BEFORE: {before_spikes_per_hour:.3f} fan spikes per hour (controller running)")
                print(f"   AFTER:  {after_spikes_per_hour:.3f} fan spikes per hour (controller running)")
                print(
                    f"   IMPROVEMENT: {spike_hour_improvement:.3f} per hour ({(spike_hour_improvement / before_spikes_per_hour * 100) if before_spikes_per_hour else 0:.1f}% reduction)"
                )
    else:
        print(f"\n💡 Use --analyze or -a flag to perform overshoot analysis")
        print(f"   Example: python feedrate_controller_comparison.py --analyze")


if __name__ == "__main__":
    main()
