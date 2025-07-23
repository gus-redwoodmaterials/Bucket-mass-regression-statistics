from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import pytz
import os
import sys
from scipy.stats import pearsonr

from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download

# Configuration
DATA_FOLDER = "data"
CSV_FILENAME = "calciner_data_july20-21_2025.csv"

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
    "kiln_weight_avg": "rc1/4420-calciner/hmi/kilnweightavg/value",
    "kiln_rpm": "rc1/4420-calciner/4420-kln-001_rpm/input",
    "kiln_weight_setpoint": "rc1/4420-calciner/hmi/4420-kln-001_weight_sp/value",
    "acme_alive": "rc1/4420-calciner/acme/acmefeedrate/acmealive",
    "acme_engineering": "rc1/4420-calciner/acme/acmefeedrate/acmeengineeringenable",
    "feed_robot_mode": "rc1/plc8-infeed_robot/hmi/feed_robot_mode/value",
}

# Try the table name your intern uses
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")

# Use a recent date range that should have data
START = pacific_tz.localize(parse("2025-07-20T08:32:00"))
END = pacific_tz.localize(parse("2025-07-21T12:00:00"))

DATA_TIMESTEP_SECONDS = 3


def run():
    # Check command line arguments for refresh flag
    refresh_data = False
    plot_data = False
    analysis_data = False

    if len(sys.argv) > 1:
        if "--refresh" in sys.argv or "-r" in sys.argv:
            refresh_data = True
            print("🔄 Refresh flag detected - will fetch fresh data from Athena")
        if "--plot" in sys.argv or "-p" in sys.argv:
            plot_data = True
            print("📊 Plot flag detected - will display plots")
        if "--analysis" in sys.argv or "-a" in sys.argv:
            analysis_data = True
            print("🔬 Analysis flag detected - will perform residual correlation analysis")

    csv_path = os.path.join(DATA_FOLDER, CSV_FILENAME)

    # Try to load from cache first
    if not refresh_data and os.path.exists(csv_path):
        print(f"Loading cached data from {csv_path}")
        df = pd.read_csv(csv_path)
        # Convert timestamp back to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print("Successfully loaded cached data!")
    else:
        print("Fetching fresh data from Athena...")
        df = athena_download.get_pivoted_athena_data(
            TAGS,
            START,
            END,
            TABLE_PATH,
            DATA_TIMESTEP_SECONDS,
        )

        # Save to CSV for future use
        print(f"Saving data to {csv_path}")
        df.to_csv(csv_path, index=False)
        print("Data cached successfully!")

    # Filter out negative Hz values
    df.loc[df["infeed_hz_actual"] < 0, "infeed_hz_actual"] = np.nan

    # Print data summary
    print("\nData Summary:")
    for col in df.columns:
        if col != "timestamp" and df[col].dtype in ["float64", "int64"]:
            if not df[col].isna().all():
                print(f"   {col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}")
            else:
                print(f"   {col}: No valid data")

    # Only plot if requested
    if plot_data:
        print("\nCreating plots...")
        mean_resid, std_resid = create_plots(df)

        # Additional residual analysis
        if mean_resid is not None and std_resid is not None:
            print(f"\n📊 Residual Statistics:")
            print(f"   Mean: {mean_resid:.4f}")
            print(f"   Standard Deviation: {std_resid:.4f}")
            print(f"   Threshold (Mean + 1σ): {mean_resid + std_resid:.4f}")

            # Find periods with high residuals
            high_resid_mask = df["bucket_mass_resid"] > (mean_resid + std_resid)
            high_resid_count = high_resid_mask.sum()
            total_points = len(df["bucket_mass_resid"].dropna())

            print(
                f"   High residual points (>1σ): {high_resid_count}/{total_points} ({100 * high_resid_count / total_points:.1f}%)"
            )

            if high_resid_count > 0:
                high_resid_times = df.loc[high_resid_mask, "timestamp"]
                print(f"   First high residual at: {high_resid_times.min()}")
                print(f"   Last high residual at: {high_resid_times.max()}")

                # Create a summary of high residual periods
                df["high_residual"] = high_resid_mask
    else:
        print("\nTip: Use --plot or -p flag to display plots")
        print("Tip: Use --refresh or -r flag to fetch fresh data from Athena")
        print("Tip: Use --analysis or -a flag to perform residual correlation analysis")

    # Perform correlation analysis if requested
    if analysis_data:
        print("\n🔬 Performing Residual Correlation Analysis...")
        perform_residual_analysis(df)

        # Add 3D correlation analysis
        print("\n📊 Performing 3D Correlation Analysis...")
        create_3d_analysis_plot(df)

    return df


def calculate_average_feed_rate(hz_signal, window_seconds=60):
    """
    Calculate moving average feed rate using a boxcar filter
    This accounts for time spent at each speed, giving a true effective feed rate
    """
    # Remove NaN values but keep track of original indices
    valid_mask = ~np.isnan(hz_signal)

    if valid_mask.sum() < 10:
        print("   ⚠️  Not enough valid conveyor data for moving average")
        return None

    # For 3-second data, calculate window size in data points
    window_size = max(1, window_seconds // 3)  # Convert seconds to data points

    # Create a copy of the signal with NaN values filled using forward/backward fill
    clean_signal = pd.Series(hz_signal).fillna(method="ffill").fillna(method="bfill").values

    # Apply moving average (boxcar filter)
    if len(clean_signal) >= window_size:
        # Use pandas rolling mean for clean handling of edges
        feed_rate_avg = pd.Series(clean_signal).rolling(window=window_size, center=True, min_periods=1).mean()
    else:
        # If not enough data for window, just return the original signal
        feed_rate_avg = pd.Series(clean_signal)

    print(f"   📊 Applied {window_seconds}s moving average filter ({window_size} data points)")
    print(f"   📊 Feed rate range: {feed_rate_avg.min():.2f} - {feed_rate_avg.max():.2f} Hz")

    return feed_rate_avg.values


def perform_residual_analysis(df):
    """
    Analyze correlation between conveyor feed rate and residual patterns
    """
    if "bucket_mass_resid" not in df.columns:
        print("   ❌ No residual data available for analysis")
        return

    if "infeed_hz_actual" not in df.columns:
        print("   ❌ No actual conveyor speed data available for analysis")
        return

    print("   🔍 Analyzing actual conveyor signal...")

    # Calculate moving average feed rate
    feed_rate_avg = calculate_average_feed_rate(df["infeed_hz_actual"], window_seconds=60)

    if feed_rate_avg is None:
        return

    print("   ✅ Calculated moving average feed rate")
    print(
        f"   📊 Feed rate values: Min={np.min(feed_rate_avg):.2f}, Max={np.max(feed_rate_avg):.2f}, Mean={np.mean(feed_rate_avg):.2f}"
    )

    # Calculate residual statistics
    residuals = df["bucket_mass_resid"].dropna()
    if len(residuals) == 0:
        print("   ❌ No valid residual data")
        return

    mean_resid = residuals.mean()
    std_resid = residuals.std()
    high_resid_threshold = mean_resid + std_resid

    # Use feed rate directly - no interpolation needed since it's already aligned with timestamps
    df_clean = df.dropna(subset=["bucket_mass_resid", "timestamp"])

    # Extract corresponding feed rates for the clean data
    feed_rate_interp = feed_rate_avg[df_clean.index]

    # Categorize feed rates
    feed_rate_median = np.median(feed_rate_interp)
    high_feed_mask = feed_rate_interp > feed_rate_median
    low_feed_mask = feed_rate_interp <= feed_rate_median

    # Calculate correlation
    if len(feed_rate_interp) == len(df_clean["bucket_mass_resid"]):
        correlation, p_value = pearsonr(feed_rate_interp, df_clean["bucket_mass_resid"])

        print("\n   📈 CORRELATION ANALYSIS:")
        print(f"   Feed Rate vs Residual Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
        if p_value < 0.05:
            print("   ✅ Statistically significant correlation!")
        else:
            print("   ⚠️  Correlation not statistically significant")

    # Analyze high residual patterns
    high_resid_mask = df_clean["bucket_mass_resid"] > high_resid_threshold

    if high_resid_mask.sum() > 0:
        high_resid_feed_rates = feed_rate_interp[high_resid_mask]

        print("\n   🎯 HIGH RESIDUAL ANALYSIS:")
        print(f"   High residuals (>{high_resid_threshold:.3f}): {high_resid_mask.sum()} points")
        print(
            f"   Feed rate during high residuals: {np.mean(high_resid_feed_rates):.2f} ± {np.std(high_resid_feed_rates):.2f}"
        )
        print(f"   Overall feed rate: {np.mean(feed_rate_interp):.2f} ± {np.std(feed_rate_interp):.2f}")

        # Check if high residuals occur more during high or low feed
        high_resid_high_feed = np.sum(high_resid_mask & high_feed_mask)
        high_resid_low_feed = np.sum(high_resid_mask & low_feed_mask)

        print("\n   📊 FEED RATE BREAKDOWN:")
        print(
            f"   High residuals during HIGH feed: {high_resid_high_feed}/{np.sum(high_feed_mask)} ({100 * high_resid_high_feed / np.sum(high_feed_mask):.1f}%)"
        )
        print(
            f"   High residuals during LOW feed:  {high_resid_low_feed}/{np.sum(low_feed_mask)} ({100 * high_resid_low_feed / np.sum(low_feed_mask):.1f}%)"
        )
        # What is this doing too?
        if high_resid_high_feed > high_resid_low_feed * 2:
            print("   🔴 Estimator performs WORSE during HIGH feed rates")
        elif high_resid_low_feed > high_resid_high_feed * 2:
            print("   🔴 Estimator performs WORSE during LOW feed rates")
        else:
            print("   ✅ No clear feed rate bias in estimator performance")

    # Create analysis plots
    create_analysis_plots(df_clean, feed_rate_interp, df["timestamp"], df["infeed_hz_actual"], high_resid_threshold)

    # Create moving average visualization plot
    create_moving_average_plot(df, feed_rate_avg)


def create_analysis_plots(df, feed_rate_interp, timestamps, raw_hz_signal, high_resid_threshold):
    """
    Create plots for residual correlation analysis
    """
    t = df["timestamp"]

    # Plot 1: Conveyor signal with peaks
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, raw_hz_signal, color="lightblue", alpha=0.7, label="Raw Conveyor Hz")
    plt.plot(df["timestamp"], feed_rate_interp, color="orange", linewidth=2, label="Moving Average Feed Rate")
    plt.ylabel("Conveyor Hz")
    plt.title("Actual Conveyor Signal with Moving Average Feed Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Feed rate vs residuals
    plt.subplot(3, 1, 2)
    plt.plot(t, df["bucket_mass_resid"], color="red", alpha=0.7, label="Residuals")
    plt.axhline(y=high_resid_threshold, color="orange", linestyle="--", label="High Residual Threshold")

    # Color points by feed rate
    plt.scatter(t, df["bucket_mass_resid"], c=feed_rate_interp, cmap="viridis", s=10, alpha=0.6, label="Feed Rate")
    plt.colorbar(label="Feed Rate (Hz)")
    plt.ylabel("Residual")
    plt.title("Residuals Colored by Feed Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Correlation scatter plot
    plt.subplot(3, 1, 3)
    plt.scatter(feed_rate_interp, df["bucket_mass_resid"], alpha=0.6, s=15)
    plt.xlabel("Feed Rate (Hz)")
    plt.ylabel("Residual")
    plt.title("Feed Rate vs Residual Correlation")
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(feed_rate_interp, df["bucket_mass_resid"], 1)
    p = np.poly1d(z)
    plt.plot(feed_rate_interp, p(feed_rate_interp), "r--", alpha=0.8, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def create_moving_average_plot(df, feed_rate_avg):
    """
    Create a plot to visualize the moving average feed rate calculation
    """
    plt.figure(figsize=(15, 8))

    # Plot 1: Original signal with moving average
    plt.subplot(2, 1, 1)
    plt.plot(
        df["timestamp"], df["infeed_hz_actual"], color="lightblue", alpha=0.7, linewidth=1, label="Raw Conveyor Hz"
    )
    plt.plot(df["timestamp"], feed_rate_avg, color="orange", linewidth=2, alpha=0.8, label="Moving Average Feed Rate")
    plt.ylabel("Conveyor Hz")
    plt.title("Moving Average Feed Rate Calculation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Zoom in on a section to see moving average detail
    # Take middle 20% of the data for detailed view
    mid_start = int(len(df) * 0.4)
    mid_end = int(len(df) * 0.6)
    df_zoom = df.iloc[mid_start:mid_end]
    feed_rate_zoom = feed_rate_avg[mid_start:mid_end]

    plt.subplot(2, 1, 2)
    plt.plot(
        df_zoom["timestamp"],
        df_zoom["infeed_hz_actual"],
        color="lightblue",
        alpha=0.7,
        linewidth=1,
        label="Raw Conveyor Hz",
    )
    plt.plot(
        df_zoom["timestamp"], feed_rate_zoom, color="orange", linewidth=2, alpha=0.8, label="Moving Average Feed Rate"
    )
    plt.ylabel("Conveyor Hz")
    plt.xlabel("Time")
    plt.title("Detailed View: How Moving Average Smooths Feed Rate Variations")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print some stats about the moving average
    print("\n   📊 MOVING AVERAGE SUMMARY:")
    print(f"   Raw signal range: {df['infeed_hz_actual'].min():.2f} - {df['infeed_hz_actual'].max():.2f} Hz")
    print(f"   Moving average range: {np.min(feed_rate_avg):.2f} - {np.max(feed_rate_avg):.2f} Hz")
    print(f"   Raw signal std dev: {df['infeed_hz_actual'].std():.2f} Hz")
    print(f"   Moving average std dev: {np.std(feed_rate_avg):.2f} Hz")
    print(f"   Smoothing factor: {df['infeed_hz_actual'].std() / np.std(feed_rate_avg):.1f}x reduction in variability")


def create_3d_analysis_plot(df):
    """
    Create a 3D plot showing the relationship between RPM, Kiln Weight, and Residuals
    """
    # Get clean data for all three variables
    required_cols = ["kiln_rpm", "kiln_weight_avg", "bucket_mass_resid"]
    df_3d = df[required_cols].dropna()

    if len(df_3d) == 0:
        print("   ❌ Not enough data for 3D analysis - missing RPM, kiln weight, or residuals")
        return

    print(f"\n   📊 3D ANALYSIS DATA:")
    print(f"   Data points available: {len(df_3d)}")
    print(f"   RPM range: {df_3d['kiln_rpm'].min():.1f} - {df_3d['kiln_rpm'].max():.1f}")
    print(f"   Kiln weight range: {df_3d['kiln_weight_avg'].min():.1f} - {df_3d['kiln_weight_avg'].max():.1f}")
    print(f"   Residual range: {df_3d['bucket_mass_resid'].min():.3f} - {df_3d['bucket_mass_resid'].max():.3f}")

    # Create 3D scatter plot
    fig = plt.figure(figsize=(15, 12))

    # Plot 1: 3D scatter plot
    ax1 = fig.add_subplot(221, projection="3d")

    # Color points by residual magnitude (absolute value)
    residual_abs = np.abs(df_3d["bucket_mass_resid"])
    scatter = ax1.scatter(
        df_3d["kiln_rpm"],
        df_3d["kiln_weight_avg"],
        df_3d["bucket_mass_resid"],
        c=residual_abs,
        cmap="viridis",
        alpha=0.6,
        s=20,
    )

    ax1.set_xlabel("Kiln RPM")
    ax1.set_ylabel("Kiln Weight (kg)")
    ax1.set_zlabel("Residual")
    ax1.set_title("3D: RPM vs Weight vs Residual\n(Color = Residual Magnitude)")

    # Add colorbar
    plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=20, label="|Residual|")

    # Plot 2: RPM vs Residual (2D projection)
    ax2 = fig.add_subplot(222)
    ax2.scatter(df_3d["kiln_rpm"], df_3d["bucket_mass_resid"], alpha=0.6, s=15)
    ax2.set_xlabel("Kiln RPM")
    ax2.set_ylabel("Residual")
    ax2.set_title("RPM vs Residual")
    ax2.grid(True, alpha=0.3)

    # Add trend line for RPM vs Residual
    z_rpm = np.polyfit(df_3d["kiln_rpm"], df_3d["bucket_mass_resid"], 1)
    p_rpm = np.poly1d(z_rpm)
    ax2.plot(
        df_3d["kiln_rpm"], p_rpm(df_3d["kiln_rpm"]), "r--", alpha=0.8, label=f"Trend: y={z_rpm[0]:.4f}x+{z_rpm[1]:.4f}"
    )
    ax2.legend()

    # Plot 3: Weight vs Residual (2D projection)
    ax3 = fig.add_subplot(223)
    ax3.scatter(df_3d["kiln_weight_avg"], df_3d["bucket_mass_resid"], alpha=0.6, s=15)
    ax3.set_xlabel("Kiln Weight (kg)")
    ax3.set_ylabel("Residual")
    ax3.set_title("Kiln Weight vs Residual")
    ax3.grid(True, alpha=0.3)

    # Add trend line for Weight vs Residual
    z_wt = np.polyfit(df_3d["kiln_weight_avg"], df_3d["bucket_mass_resid"], 1)
    p_wt = np.poly1d(z_wt)
    ax3.plot(
        df_3d["kiln_weight_avg"],
        p_wt(df_3d["kiln_weight_avg"]),
        "r--",
        alpha=0.8,
        label=f"Trend: y={z_wt[0]:.4f}x+{z_wt[1]:.4f}",
    )
    ax3.legend()

    # Plot 4: RPM vs Weight (colored by residual)
    ax4 = fig.add_subplot(224)
    scatter2 = ax4.scatter(
        df_3d["kiln_rpm"], df_3d["kiln_weight_avg"], c=df_3d["bucket_mass_resid"], cmap="RdBu_r", alpha=0.6, s=20
    )
    ax4.set_xlabel("Kiln RPM")
    ax4.set_ylabel("Kiln Weight (kg)")
    ax4.set_title("RPM vs Weight\n(Color = Residual)")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label="Residual")

    plt.tight_layout()
    plt.show()

    # Calculate and print correlations
    print(f"\n   📈 CORRELATION MATRIX:")
    corr_rpm_resid = np.corrcoef(df_3d["kiln_rpm"], df_3d["bucket_mass_resid"])[0, 1]
    corr_weight_resid = np.corrcoef(df_3d["kiln_weight_avg"], df_3d["bucket_mass_resid"])[0, 1]
    corr_rpm_weight = np.corrcoef(df_3d["kiln_rpm"], df_3d["kiln_weight_avg"])[0, 1]

    print(f"   RPM ↔ Residual:     {corr_rpm_resid:.4f}")
    print(f"   Weight ↔ Residual:  {corr_weight_resid:.4f}")
    print(f"   RPM ↔ Weight:       {corr_rpm_weight:.4f}")

    # Statistical significance tests
    from scipy.stats import pearsonr

    _, p_rpm_resid = pearsonr(df_3d["kiln_rpm"], df_3d["bucket_mass_resid"])
    _, p_weight_resid = pearsonr(df_3d["kiln_weight_avg"], df_3d["bucket_mass_resid"])
    _, p_rpm_weight = pearsonr(df_3d["kiln_rpm"], df_3d["kiln_weight_avg"])

    print(f"\n   🧪 STATISTICAL SIGNIFICANCE (p-values):")
    print(
        f"   RPM ↔ Residual:     p={p_rpm_resid:.4f} {'✅ Significant' if p_rpm_resid < 0.05 else '❌ Not significant'}"
    )
    print(
        f"   Weight ↔ Residual:  p={p_weight_resid:.4f} {'✅ Significant' if p_weight_resid < 0.05 else '❌ Not significant'}"
    )
    print(
        f"   RPM ↔ Weight:       p={p_rpm_weight:.4f} {'✅ Significant' if p_rpm_weight < 0.05 else '❌ Not significant'}"
    )

    return df_3d


def create_plots(df):
    """Create separate plots for better visibility"""
    t = df["timestamp"]

    # Plot 1: Bucket Mass Data
    plt.figure(figsize=(14, 6))
    if "bucket_mass" in df.columns:
        plt.plot(t, df["bucket_mass"], color="blue", linewidth=2, label="Bucket Mass")
    if "bucket_mass_regress" in df.columns:
        plt.plot(t, df["bucket_mass_regress"], color="red", linewidth=1, alpha=0.7, label="Bucket Mass Regression")

    plt.ylabel("Mass")
    plt.xlabel("Time")
    plt.title("Bucket Mass Analysis (July 20-21, 2025)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Kiln Operations
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()  # Create twin axis for different scales

    if "kiln_weight_avg" in df.columns:
        ax1.plot(t, df["kiln_weight_avg"], color="green", linewidth=2, label="Kiln Weight Avg")
    if "kiln_weight_setpoint" in df.columns:
        ax1.plot(
            t, df["kiln_weight_setpoint"], color="darkgreen", linewidth=1, linestyle="--", label="Kiln Weight Setpoint"
        )

    if "kiln_rpm" in df.columns:
        ax2.plot(t, df["kiln_rpm"], color="orange", linewidth=2, label="Kiln RPM")

    ax1.set_ylabel("Weight", color="green")
    ax2.set_ylabel("RPM", color="orange")
    ax1.set_xlabel("Time")
    ax1.set_title("Kiln Operations")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 3: Conveyor and Feed Systems
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    if "infeed_hz_actual" in df.columns:
        ax1.plot(t, df["infeed_hz_actual"], color="purple", linewidth=2, label="Actual Conveyor Speed (Hz)")
    if "infeed_hz_sp" in df.columns:
        ax1.plot(
            t, df["infeed_hz_sp"], color="magenta", linewidth=2, alpha=0.7, linestyle="--", label="Setpoint Conveyor Hz"
        )

    if "cycle_time_complete" in df.columns:
        ax2.plot(t, df["cycle_time_complete"], color="brown", linewidth=2, label="Cycle Time Complete")
    if "large_module_cycle_time" in df.columns:
        ax2.plot(
            t, df["large_module_cycle_time"], color="chocolate", linewidth=1, alpha=0.7, label="Large Module Cycle Time"
        )

    ax1.set_ylabel("Speed (Hz)", color="purple")
    ax2.set_ylabel("Time", color="brown")
    ax1.set_xlabel("Time")
    ax1.set_title("Conveyor and Cycle Times")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 4: System Status
    plt.figure(figsize=(14, 6))
    if "acme_alive" in df.columns:
        plt.plot(t, df["acme_alive"], color="red", linewidth=2, label="ACME Alive", marker="o", markersize=2)
    if "acme_engineering" in df.columns:
        plt.plot(
            t, df["acme_engineering"], color="blue", linewidth=2, label="ACME Engineering", marker="s", markersize=2
        )
    if "feed_robot_mode" in df.columns:
        plt.plot(
            t, df["feed_robot_mode"], color="green", linewidth=2, label="Feed Robot Mode", marker="^", markersize=2
        )

    plt.ylabel("Status/Mode")
    plt.xlabel("Time")
    plt.title("System Status Indicators")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 5: Residual Analysis
    if "bucket_mass_resid" in df.columns:
        residuals = df["bucket_mass_resid"].dropna()
        if len(residuals) > 0:
            mean_resid = residuals.mean()
            std_resid = residuals.std()

            plt.figure(figsize=(14, 8))

            # Top subplot: Residuals over time
            plt.subplot(2, 1, 1)
            plt.plot(t, df["bucket_mass_resid"], color="red", linewidth=1, alpha=0.7, label="Residuals")
            plt.axhline(y=mean_resid, color="black", linestyle="--", label=f"Mean: {mean_resid:.3f}")
            plt.axhline(
                y=mean_resid + std_resid, color="orange", linestyle="--", label=f"+1 STD: {mean_resid + std_resid:.3f}"
            )
            plt.axhline(
                y=mean_resid - std_resid, color="orange", linestyle="--", label=f"-1 STD: {mean_resid - std_resid:.3f}"
            )

            # Highlight points > 1 std above mean
            high_resid_mask = df["bucket_mass_resid"] > (mean_resid + std_resid)
            if high_resid_mask.any():
                plt.scatter(
                    t[high_resid_mask],
                    df["bucket_mass_resid"][high_resid_mask],
                    color="red",
                    s=20,
                    zorder=5,
                    label="High Residuals (>1σ)",
                )

            plt.ylabel("Residual")
            plt.title("Bucket Mass Residual Analysis")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # Bottom subplot: Histogram of residuals
            plt.subplot(2, 1, 2)
            plt.hist(residuals, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
            plt.axvline(x=mean_resid, color="black", linestyle="--", label=f"Mean: {mean_resid:.3f}")
            plt.axvline(x=mean_resid + std_resid, color="orange", linestyle="--", label="+1 STD")
            plt.axvline(x=mean_resid - std_resid, color="orange", linestyle="--", label="-1 STD")
            plt.xlabel("Residual Value")
            plt.ylabel("Frequency")
            plt.title("Distribution of Residuals")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            return mean_resid, std_resid
        else:
            print("No valid residual data found")
            return None, None
    else:
        print("No residual column found in data")
        return None, None


if __name__ == "__main__":
    # Run the function and get the dataframe for analysis
    df = run()

    # Now you can do analysis with the dataframe
    print("\nReady for analysis! The dataframe 'df' is available.")
    print("   Example analysis commands:")
    print("   - df.head() - view first few rows")
    print("   - df.describe() - statistical summary")
    print("   - df['bucket_mass'].plot() - quick plot")
    print("   - df.corr() - correlation matrix")
