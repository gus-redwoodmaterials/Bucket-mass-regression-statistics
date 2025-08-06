import pandas as pd
import numpy as np
import pytz
import os
import sys
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download
import matplotlib.pyplot as plt
import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


DATA_FOLDER = "data"
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")
DATA_TIMESTEP_SECONDS = 1


if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

TAGS = {
    "Motor amps": "RC1/4420-Calciner/4420-KLN-001-MTR-001_Current/Input".lower(),
    "Small Mod Feed": "RC1/4420-Calciner/4420-CVR-001/Status/Speed_Feedback_Hz".lower(),
    "Robot ON": "RC1/PLC8-Infeed_Robot/HMI/Feed_Robot_Mode/Value".lower(),
    "RPM": "RC1/4420-Calciner/4420-KLN-001_RPM/Input".lower(),
    "Kiln Weight": "RC1/4420-Calciner/4420-WT-0007_Alarm/Input".lower(),
    "Large Mod Feed": "RC1/4420-Calciner/Module_Loading/HMI/Cycle_Time_Complete/Value".lower(),
    "N2 Cons": "RC1/4420-Calciner/4420-FT-0001_Vol_Flow_Ave/Input".lower(),
    "Zone 1 Temp": "RC1/4420-Calciner/4420-TE-7210-B_AI/Input_Scaled".lower(),
}


def load_data(start_date_str, end_date_str, description="data"):
    start = pacific_tz.localize(parse(start_date_str))
    end = pacific_tz.localize(parse(end_date_str))

    start_str = start.strftime("%Y%m%d_%H%M")
    end_str = end.strftime("%Y%m%d_%H%M")
    csv_filename = f"athena_{description}_{start_str}_to_{end_str}.csv"
    csv_avg_filename = f"athena_{description}_{start_str}_to_{end_str}_avg.csv"
    csv_path = os.path.join(DATA_FOLDER, csv_filename)
    csv_avg_path = os.path.join(DATA_FOLDER, csv_avg_filename)

    refresh_data = "--refresh" in sys.argv or "-r" in sys.argv

    if not refresh_data and os.path.exists(csv_path):
        print(f"Loading from cache: {csv_filename}")
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Check if averaged data exists
        if os.path.exists(csv_avg_path):
            print(f"Loading averaged data from cache: {csv_avg_filename}")
            df_avg = pd.read_csv(csv_avg_path)
            df_avg["timestamp"] = pd.to_datetime(df_avg["timestamp"])
            return df, df_avg
        else:
            print("Calculating per-revolution averages...")
            df_avg = make_rpm_rolling_avg(df)
            df_avg.to_csv(csv_avg_path, index=False)
            print(f"Cached averaged data: {csv_avg_filename}")
            return df, df_avg
    else:
        print(f"Fetching from Athena: {start_str} to {end_str}")
        try:
            df = athena_download.get_pivoted_athena_data(
                TAGS,
                start,
                end,
                TABLE_PATH,
                DATA_TIMESTEP_SECONDS,
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.to_csv(csv_path, index=False)
            print(f"Cached: {csv_filename}")

            # Calculate and cache averaged data
            print("Calculating per-revolution averages...")
            df_avg = make_rpm_rolling_avg(df)
            df_avg.to_csv(csv_avg_path, index=False)
            print(f"Cached averaged data: {csv_avg_filename}")

            return df, df_avg
        except KeyError as e:
            print(f"Error fetching data: {e}")
            print("This usually means no data was returned for the specified tags and time range")
            print(f"Tags being queried: {list(TAGS.keys())}")
            return pd.DataFrame(), pd.DataFrame()  # Return empty dataframes


def make_rpm_rolling_avg(df):
    """
    Get rolling avg (per single rev) for all sensor values
    """
    t = df["timestamp"].to_numpy()
    rpm = df["RPM"].to_numpy()
    angle = 0  # degrees

    t_avg = []
    wt_avg = []
    rpm_avg = []
    motor_amps_avg = []
    small_mod_feed_avg = []
    robot_on_avg = []
    large_mod_feed_avg = []
    n2_cons_avg = []
    zone1_temp_avg = []

    revs_complete = 0
    wt_rev = 0
    rpm_rev = 0
    motor_amps_rev = 0
    small_mod_feed_rev = 0
    robot_on_rev = 0
    large_mod_feed_rev = 0
    n2_cons_rev = 0
    zone1_temp_rev = 0
    num_pts_rev = 0

    for i in range(1, len(rpm)):
        angle += rpm[i] * (t[i] - t[i - 1]) / np.timedelta64(1, "s") / 60 * 360

        wt_rev += df.iloc[i, df.columns.get_loc("Kiln Weight")]
        rpm_rev += rpm[i]
        motor_amps_rev += df.iloc[i, df.columns.get_loc("Motor amps")]
        small_mod_feed_rev += df.iloc[i, df.columns.get_loc("Small Mod Feed")]
        robot_on_rev += df.iloc[i, df.columns.get_loc("Robot ON")]
        large_mod_feed_rev += df.iloc[i, df.columns.get_loc("Large Mod Feed")]
        n2_cons_rev += df.iloc[i, df.columns.get_loc("N2 Cons")]
        zone1_temp_rev += df.iloc[i, df.columns.get_loc("Zone 1 Temp")]
        num_pts_rev += 1

        if angle >= 360 * (revs_complete + 1):
            revs_complete += 1
            t_avg.append(t[i])
            rpm_avg.append(rpm_rev / num_pts_rev)
            wt_avg.append(wt_rev / num_pts_rev)
            motor_amps_avg.append(motor_amps_rev / num_pts_rev)
            small_mod_feed_avg.append(small_mod_feed_rev / num_pts_rev)
            robot_on_avg.append(robot_on_rev / num_pts_rev)
            large_mod_feed_avg.append(large_mod_feed_rev / num_pts_rev)
            n2_cons_avg.append(n2_cons_rev / num_pts_rev)
            zone1_temp_avg.append(zone1_temp_rev / num_pts_rev)

            wt_rev = 0
            rpm_rev = 0
            motor_amps_rev = 0
            small_mod_feed_rev = 0
            robot_on_rev = 0
            large_mod_feed_rev = 0
            n2_cons_rev = 0
            zone1_temp_rev = 0
            num_pts_rev = 0

    df_avg = pd.DataFrame(
        {
            "timestamp": np.array(t_avg),
            "Kiln Weight": np.array(wt_avg),
            "RPM": np.array(rpm_avg),
            "Motor amps": np.array(motor_amps_avg),
            "Small Mod Feed": np.array(small_mod_feed_avg),
            "Robot ON": np.array(robot_on_avg),
            "Large Mod Feed": np.array(large_mod_feed_avg),
            "N2 Cons": np.array(n2_cons_avg),
            "Zone 1 Temp": np.array(zone1_temp_avg),
        }
    )

    return df_avg


def find_motor_spikes(df, threshold=40, time_offset=30):
    """
    Find periods where motor amps stay above threshold for at least 3 continuous seconds.
    Returns list of spike start timestamps.
    """
    spike_offset = pd.Timedelta(seconds=60 * time_offset)  # 2 minute offset
    df = df.copy()

    # Create boolean mask for values above threshold
    above_threshold = df["Motor amps"] > threshold

    # Find continuous periods above threshold
    spike_starts = []
    in_spike = False
    spike_start_time = None

    for i, (timestamp, is_above) in enumerate(zip(df["timestamp"], above_threshold)):
        if is_above and not in_spike:
            # Start of potential spike
            spike_start_time = timestamp
            spike_start_idx = i
            in_spike = True
        elif not is_above and in_spike:
            # End of spike - check if it lasted at least 3 seconds
            spike_duration = timestamp - spike_start_time
            if spike_duration >= pd.Timedelta(seconds=3):
                spike_starts.append(spike_start_time)
            in_spike = False
        elif i == len(df) - 1 and in_spike:
            # Handle case where spike continues to end of data
            spike_duration = timestamp - spike_start_time
            if spike_duration >= pd.Timedelta(seconds=3):
                spike_starts.append(spike_start_time)

    # Filter out spikes that are too close together
    # filtered_spikes = spike_starts.copy()
    # Uncomment the following lines if you want to filter spikes based on a minimum offset
    filtered_spikes = []
    for spike_time in spike_starts:
        if not filtered_spikes or all(abs(spike_time - prev_spike) >= spike_offset for prev_spike in filtered_spikes):
            filtered_spikes.append(spike_time)

    return filtered_spikes


def plot_spike(spike_time, df, df_avg=None, look_back_window=20, show_small_mod_feed=True):
    mask = (df["timestamp"] >= spike_time - pd.Timedelta(seconds=look_back_window * 60)) & (
        df["timestamp"] <= spike_time + pd.Timedelta(seconds=3 * 60)
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Motor amps (both normal and averaged if available)
    ax1_twin = ax1.twinx()

    # Plot normal motor amps
    ax1.plot(
        df["timestamp"][mask],
        df["Motor amps"][mask],
        color="orange",
        linewidth=1.5,
        alpha=0.7,
        label="Motor Amps (Normal)",
    )

    # Plot averaged motor amps if df_avg is provided
    if df_avg is not None:
        mask_avg = (df_avg["timestamp"] >= spike_time - pd.Timedelta(seconds=look_back_window * 60)) & (
            df_avg["timestamp"] <= spike_time + pd.Timedelta(seconds=3 * 60)
        )
        ax1.plot(
            df_avg["timestamp"][mask_avg],
            df_avg["Motor amps"][mask_avg],
            color="red",
            linewidth=2.5,
            label="Motor Amps (Averaged)",
        )

    # Add threshold line
    ax1.axhline(y=40, color="black", linestyle="--", linewidth=2, label="40A Threshold")

    if show_small_mod_feed:
        ax1_twin.plot(
            df["timestamp"][mask], df["Small Mod Feed"][mask], color="purple", linewidth=1.5, label="Small Mod Feed"
        )
        ax1_twin.set_ylabel("Small Mod Feed (Hz)", color="purple")
    else:
        ax1_twin.set_ylabel("")

    ax1.set_ylabel("Motor Amps", color="orange")
    ax1.set_title("Motor Amps Comparison{}".format(" and Small Mod Feed" if show_small_mod_feed else ""))
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Kiln Weight and RPM (dual y-axis)
    ax2_twin = ax2.twinx()
    ax2.plot(df["timestamp"][mask], df["Kiln Weight"][mask], color="green", linewidth=1.5, label="Kiln Weight")
    ax2_twin.plot(df["timestamp"][mask], df["RPM"][mask], color="red", linewidth=1.5, label="RPM")
    ax2.set_ylabel("Kiln Weight (kg)", color="green")
    ax2_twin.set_ylabel("RPM", color="red")
    ax2.set_title("Kiln Weight and RPM")
    ax2.grid(True, alpha=0.3)

    # Plot 3: N2 Consumption and Zone 1 Temp (dual y-axis)
    ax3_twin = ax3.twinx()
    ax3.plot(df["timestamp"][mask], df["N2 Cons"][mask], color="blue", linewidth=1.5, label="N2 Consumption")
    ax3_twin.plot(df["timestamp"][mask], df["Zone 1 Temp"][mask], color="olive", linewidth=1.5, label="Zone 1 Temp")
    ax3.set_ylabel("N2 Consumption", color="blue")
    ax3_twin.set_ylabel("Zone 1 Temp (°C)", color="olive")
    ax3.set_title("N2 Consumption and Zone 1 Temperature")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def robot_analysis(spike_times, df):
    if "Robot ON" not in df.columns or len(spike_times) == 0:
        return 0, 0

    robot_on_spike = 0

    # For per-revolution data, we need to adjust the threshold since values are averaged
    filtered_df = df.loc[df["Robot ON"] > 0.5].copy()  # Adjusted threshold for averaged data

    if len(filtered_df) == 0:
        return 0, 0

    # This creates a new group ID every time the condition changes
    filtered_df = filtered_df.copy()
    filtered_df["segment"] = (~filtered_df.index.to_series().diff().eq(1)).cumsum()

    # Split into list of DataFrames
    robot_segment_dfs = [group for name, group in filtered_df.groupby("segment")]

    total_robot_segments = 0
    for segment_df in robot_segment_dfs:
        # For per-revolution data, we need fewer points to constitute a valid segment
        if len(segment_df) < 3:  # At least 3 revolutions
            continue
        total_robot_segments += 1

    for spike_time in spike_times:
        # Filter dataframe by timestamp and check Robot ON value
        robot_value = df[df["timestamp"] == spike_time]["Robot ON"]
        if not robot_value.empty and robot_value.iloc[0] > 0.5:  # Adjusted threshold for averaged data
            robot_on_spike += 1

    # Calculate the percentage of spikes where the robot was on and the percentage of times the robot is on that it spikes
    if total_robot_segments == 0:
        return robot_on_spike / len(spike_times), 0
    return robot_on_spike / len(spike_times), robot_on_spike / total_robot_segments


def correlation_analysis(spike_times, df):
    """
    Analyze correlation between spike times and all tracked variables
    """
    if len(spike_times) == 0:
        print("No spikes found for correlation analysis")
        return

    # Create a binary spike indicator column
    df_corr = df.copy()
    df_corr["is_spike"] = 0

    # Mark spike times
    for spike_time in spike_times:
        mask = df_corr["timestamp"] == spike_time
        df_corr.loc[mask, "is_spike"] = 1

    # Get numeric columns for correlation
    numeric_cols = [
        "Motor amps",
        "Small Mod Feed",
        "Robot ON",
        "RPM",
        "Kiln Weight",
        "Large Mod Feed",
        "N2 Cons",
        "Zone 1 Temp",
    ]

    # Filter to only existing columns
    available_cols = [col for col in numeric_cols if col in df_corr.columns]

    print("\n=== CORRELATION ANALYSIS ===")
    print(f"Analyzing {len(spike_times)} spikes across {len(df_corr)} data points")
    print(f"Spike rate: {len(spike_times) / len(df_corr) * 100:.2f}%\n")

    correlations = {}
    for col in available_cols:
        corr = df_corr["is_spike"].corr(df_corr[col])
        correlations[col] = corr
        print(f"{col:15}: {corr:6.3f}")

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n=== TOP CORRELATIONS (by absolute value) ===")
    for var, corr in sorted_corr:
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"{var:15}: {corr:6.3f} ({strength} {direction})")

    # Statistical summary around spikes
    print(f"\n=== VARIABLE VALUES DURING SPIKES ===")
    spike_data = []
    for spike_time in spike_times:
        spike_row = df_corr[df_corr["timestamp"] == spike_time]
        if not spike_row.empty:
            spike_data.append(spike_row.iloc[0])

    if spike_data:
        spike_df = pd.DataFrame(spike_data)
        print(f"{'Variable':<15} {'During Spikes':<15} {'Overall Mean':<15} {'Difference':<15}")
        print("-" * 65)

        for col in available_cols:
            if col in spike_df.columns:
                spike_mean = spike_df[col].mean()
                overall_mean = df_corr[col].mean()
                diff = spike_mean - overall_mean
                print(f"{col:<15} {spike_mean:<15.2f} {overall_mean:<15.2f} {diff:<15.2f}")


def run():
    start_utc = "2025-06-15 00:00:00"
    stop_utc = "2025-08-04 00:00:00"

    df = load_data(start_utc, stop_utc)
    find_motor_spikes(df)
    df.to_csv(os.path.join(DATA_FOLDER, "motor_spike_times.csv"), index=False)


#     feed_df = pd.read_csv("scout_full_Xy.csv")

#     df, df_avg = load_data(start_date, end_date, "test")
#     print(f"Shape: {df.shape}")
#     print(df.head())

#     print(f"Shape after rpm rolling avg: {df_avg.shape}")
#     print(df_avg.head())

#     normal_spike_windows = find_motor_spikes(df, threshold=40, time_offset=30)
#     df["is_spike"] = 0
#     for spike_time in normal_spike_windows:
#         df.loc[df["timestamp"] == spike_time, "is_spike"] = 1
#     # averaged_spike_windows = find_motor_spikes(df_avg, threshold=40, time_offset=30)
#     print(f"Found {len(normal_spike_windows)} spike windows:")
#     if normal_spike_windows:
#         for spike_time in normal_spike_windows:
#             # if not robot_value.empty and robot_value.iloc[0] > 0:
#             # plot_spike(spike_time, df_avg, show_small_mod_feed=True)
#             print(f"  {spike_time} - Avg: {df[df['timestamp'] == spike_time]['Motor amps'].mean():.2f} A")
#             # plot_spike(spike_time, df, df_avg, show_small_mod_feed=False)  # Pass both dataframes

#     robo_helper = robot_analysis(normal_spike_windows, df)
#     print(f"\nRobot ON percentage during spikes: {robo_helper[0] * 100:.2f}%")
#     print(f"Percentage if times the robot spikes when it is run: {robo_helper[1] * 100:.2f}%")

#     feed_df[["window_start_utc", "window_end_utc"]] = feed_df[["window_start_utc", "window_end_utc"]].apply(
#         pd.to_datetime
#     )
#     # make the timestamps tz-aware and convert to your kiln’s timezone
#     feed_df[["window_start_utc", "window_end_utc"]] = feed_df[["window_start_utc", "window_end_utc"]].apply(
#         lambda col: col.dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
#     )

#     feed_cols = [c for c in feed_df.columns if c.startswith("mean_")]
#     # Run correlation analysis
#     # correlation_analysis(normal_spike_windows, df_avg)

#     # 2.  Build an IntervalIndex so we can *map any kiln timestamp* → *feed window*
#     interval_index = pd.IntervalIndex.from_arrays(feed_df["window_start_utc"], feed_df["window_end_utc"], closed="left")
#     feed_df = feed_df.set_index(interval_index)  # rows now keyed by interval
#     # ──────────────────────────────────────────────────────────────────────────────
#     # 3.  Map feed chemistry onto every row in your kiln-current dataframe `df`
#     #     • df.index must already be a tz-aware DatetimeIndex (same timezone!)
#     #     • interval_index.get_indexer(df.index) returns, for each sample,
#     #       the row-number of the feed window it falls into (-1 → no window)

#     idx = interval_index.get_indexer(df.index)
#     # rows with -1 mean “no feed info” (gap between windows); set to NaN
#     feed_features = feed_df.iloc[idx].reset_index(drop=True)
#     feed_features.loc[idx == -1, feed_cols] = np.nan

#     # attach to the kiln dataframe
#     df = pd.concat([df, feed_features[feed_cols]], axis=1)
#     print(df.head())
#     # ──────────────────────────────────────────────────────────────────────────────
#     # 4.  Run the correlation analysis with lag optimization
#     df_final, lag_analysis = test_lag_correlations_and_analyze(df, feed_cols)

#     print(f"\n=== ANALYSIS COMPLETE ===")
#     print(f"Data saved with feed chemistry features and optimal lag applied")
#     print(f"You can now use df_final for further analysis or machine learning")


# def test_lag_correlations_and_analyze(df, feed_cols):
#     # ──────────────────────────────────────────────────────────────────────────────
#     # 5.  Test different lag times to find optimal correlation
#     def test_lag_correlations(df, feed_cols, spike_col="is_spike", max_lag_min=20):
#         """Test different lag times to find optimal correlation with spikes"""
#         lag_results = []

#         for lag_min in range(0, max_lag_min + 1, 2):  # Test every 2 minutes
#             # Create temporary lagged features
#             df_temp = df.copy()
#             lag_seconds = int(lag_min * 60 / DATA_TIMESTEP_SECONDS)
#             df_temp[feed_cols] = df_temp[feed_cols].shift(lag_seconds)

#             # Calculate correlations
#             corr = df_temp[feed_cols].corrwith(df_temp[spike_col])
#             max_corr = corr.abs().max()
#             best_feature = corr.abs().idxmax()
#             print(corr.head())

#             lag_results.append(
#                 {
#                     "lag_minutes": lag_min,
#                     "max_correlation": max_corr,
#                     "best_feature": best_feature,
#                     "correlation_value": corr[best_feature],
#                 }
#             )

#         return pd.DataFrame(lag_results)

#     print("\n=== TESTING DIFFERENT LAG TIMES ===")
#     lag_results = test_lag_correlations(df, feed_cols)
#     print(lag_results.sort_values("max_correlation", ascending=False).head(10))

#     # Use the optimal lag time
#     optimal_lag = lag_results.loc[lag_results["max_correlation"].idxmax(), "lag_minutes"]
#     print(f"\nOptimal lag time: {optimal_lag} minutes")

#     # Apply optimal lag
#     LAG_MIN = optimal_lag
#     df[feed_cols] = df[feed_cols].shift(int(LAG_MIN * 60 / DATA_TIMESTEP_SECONDS))

#     # ──────────────────────────────────────────────────────────────────────────────
#     # 6.  Detailed correlation analysis with optimal lag
#     print(f"\n=== CORRELATION ANALYSIS (LAG = {LAG_MIN} min) ===")
#     corr = df[feed_cols].corrwith(df["is_spike"])
#     corr_sorted = corr.sort_values(ascending=False, key=abs)

#     print("Top correlations (absolute value):")
#     for feature, correlation in corr_sorted.head(15).items():
#         strength = "Strong" if abs(correlation) > 0.3 else "Moderate" if abs(correlation) > 0.1 else "Weak"
#         direction = "↑ Higher" if correlation > 0 else "↓ Lower"
#         print(f"{feature:<30}: {correlation:7.4f} ({strength}) - {direction} levels = more spikes")

#     # ──────────────────────────────────────────────────────────────────────────────
#     # 7.  Statistical analysis of feed composition during spikes
#     print(f"\n=== FEED COMPOSITION DURING SPIKES ===")

#     spike_data = df[df["is_spike"] == 1][feed_cols].dropna()
#     normal_data = df[df["is_spike"] == 0][feed_cols].dropna()

#     if len(spike_data) > 0 and len(normal_data) > 0:
#         print(f"Analyzing {len(spike_data)} spike periods vs {len(normal_data)} normal periods")
#         print(f"{'Feed Component':<30} {'During Spikes':<15} {'Normal Times':<15} {'Difference':<15} {'% Change':<10}")
#         print("-" * 85)

#         for col in feed_cols:
#             if col in spike_data.columns and not spike_data[col].empty and not normal_data[col].empty:
#                 spike_mean = spike_data[col].mean()
#                 normal_mean = normal_data[col].mean()
#                 diff = spike_mean - normal_mean
#                 pct_change = (diff / normal_mean * 100) if normal_mean != 0 else 0

#                 print(f"{col:<30} {spike_mean:<15.4f} {normal_mean:<15.4f} {diff:<15.4f} {pct_change:<10.2f}%")

#     # ──────────────────────────────────────────────────────────────────────────────
#     # 8.  Identify problematic battery types
#     print(f"\n=== PROBLEMATIC BATTERY COMPOSITIONS ===")

#     # Find the most correlated features
#     top_correlations = corr_sorted.head(10)

#     for feature, correlation in top_correlations.items():
#         if abs(correlation) > 0.05:  # Only show meaningful correlations
#             print(f"\n{feature}:")
#             print(f"  Correlation with spikes: {correlation:.4f}")

#             # Show distribution during spikes vs normal
#             if not spike_data.empty and feature in spike_data.columns:
#                 spike_values = spike_data[feature].dropna()
#                 normal_values = normal_data[feature].dropna()

#                 if len(spike_values) > 0 and len(normal_values) > 0:
#                     print(f"  During spikes - Mean: {spike_values.mean():.4f}, Std: {spike_values.std():.4f}")
#                     print(f"  Normal times - Mean: {normal_values.mean():.4f}, Std: {normal_values.std():.4f}")

#                     # Percentile analysis
#                     if correlation > 0:
#                         threshold = normal_values.quantile(0.75)
#                         print(f"  Risk threshold (75th percentile): {threshold:.4f}")
#                     else:
#                         threshold = normal_values.quantile(0.25)
#                         print(f"  Risk threshold (25th percentile): {threshold:.4f}")

#     print(f"\n=== SUMMARY ===")
#     print(f"• Optimal lag time: {LAG_MIN} minutes")
#     print(f"• Total spikes analyzed: {df['is_spike'].sum()}")
#     print(f"• Feed data coverage: {(~df[feed_cols].isna().all(axis=1)).sum()}/{len(df)} timestamps")

#     return df, lag_results


if __name__ == "__main__":
    run()
