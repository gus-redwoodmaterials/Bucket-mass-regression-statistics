import pandas as pd
import numpy as np
import pytz
import os
import sys
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download
import matplotlib.pyplot as plt
import warnings

# Calculate R^2 for each period (linear regression)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    "RPM": "RC1/4420-Calciner/4420-KLN-001_RPM/Input".lower(),
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
    rpm_avg = []
    motor_amps_avg = []

    revs_complete = 0
    rpm_rev = 0
    motor_amps_rev = 0
    num_pts_rev = 0

    for i in range(1, len(rpm)):
        angle += rpm[i] * (t[i] - t[i - 1]) / np.timedelta64(1, "s") / 60 * 360

        rpm_rev += rpm[i]
        motor_amps_rev += df.iloc[i, df.columns.get_loc("Motor amps")]
        num_pts_rev += 1

        if angle >= 360 * (revs_complete + 1):
            revs_complete += 1
            t_avg.append(t[i])
            rpm_avg.append(rpm_rev / num_pts_rev)
            motor_amps_avg.append(motor_amps_rev / num_pts_rev)

            rpm_rev = 0
            motor_amps_rev = 0
            num_pts_rev = 0

    df_avg = pd.DataFrame(
        {
            "timestamp": np.array(t_avg),
            "RPM": np.array(rpm_avg),
            "Motor amps": np.array(motor_amps_avg),
        }
    )

    return df_avg


def plot_avg_motor_amps(df_avg):
    """
    Plot the average motor amps over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_avg["timestamp"], df_avg["Motor amps"], label="Avg Motor Amps", color="blue")
    plt.xlabel("Timestamp")
    plt.ylabel("Average Motor Amps")
    plt.title("Average Motor Amps Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()

    # Add vertical lines for every Wednesday after Jan 1, 2025
    # Jan 1, 2025 is a Wednesday, so start from Jan 8, 2025
    min_time = df_avg["timestamp"].min()
    max_time = df_avg["timestamp"].max()
    wednesday = pd.Timestamp("2025-01-08", tz="UTC")
    while wednesday <= max_time:
        plt.axvline(
            wednesday,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Wednesday" if wednesday == pd.Timestamp("2025-01-08") else None,
        )
        wednesday += pd.Timedelta(days=7)

    # Add a big solid red line at January 22, 2025 noon
    jan22 = pd.Timestamp("2025-01-22 12:00", tz="UTC")
    plt.axvline(jan22, color="red", linestyle="-", linewidth=3, label="Jan 22 Noon")

    plt.tight_layout()
    plt.show()

    # 2D histogram of RPM vs Avg Motor Amps, before and after Jan 22, 2025 noon
    jan22 = pd.Timestamp("2025-01-22 12:00", tz="UTC")
    before = df_avg[df_avg["timestamp"] < jan22]
    after = df_avg[df_avg["timestamp"] >= jan22]

    def calc_r2(x, y):
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        if len(x) < 2:
            return np.nan
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        return r2_score(y, y_pred)

    r2_before = calc_r2(before["RPM"], before["Motor amps"])
    r2_after = calc_r2(after["RPM"], after["Motor amps"])
    print(f"R^2 before Jan 22, 2025 noon: {r2_before:.4f}")
    print(f"R^2 after Jan 22, 2025 noon: {r2_after:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    h1 = axes[0].hist2d(before["RPM"], before["Motor amps"], bins=50, cmap="Blues")
    axes[0].set_title(f"Before Jan 22, 2025 Noon\n$R^2$ = {r2_before:.3f}")
    axes[0].set_xlabel("RPM")
    axes[0].set_ylabel("Avg Motor Amps")
    axes[0].set_ylim(15, 40)
    plt.colorbar(h1[3], ax=axes[0], label="Counts")

    h2 = axes[1].hist2d(after["RPM"], after["Motor amps"], bins=50, cmap="Reds")
    axes[1].set_title(f"After Jan 22, 2025 Noon\n$R^2$ = {r2_after:.3f}")
    axes[1].set_xlabel("RPM")
    axes[1].set_ylabel("")
    axes[1].set_ylim(15, 40)
    plt.colorbar(h2[3], ax=axes[1], label="Counts")

    plt.suptitle("2D Histogram: RPM vs Avg Motor Amps")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run():
    # Load data
    start_date_str = "2024-11-01 00:00"
    end_date_str = "2025-03-31 23:59"
    df, df_avg = load_data(start_date_str, end_date_str)
    plot_avg_motor_amps(df_avg)


if __name__ == "__main__":
    run()
