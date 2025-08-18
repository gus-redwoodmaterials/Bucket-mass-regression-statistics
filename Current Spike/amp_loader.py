import pandas as pd
import numpy as np
import pytz
import os
import sys
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt

# Suppress all FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)


DATA_FOLDER = "data"
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")
DATA_TIMESTEP_SECONDS = 1


if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

TAGS = {
    "motor_amps": "RC1/4420-Calciner/4420-KLN-001-MTR-001_Current/Input".lower(),
    "small_mod_feed": "RC1/4420-Calciner/4420-CVR-001/Status/Speed_Feedback_Hz".lower(),  # 16 hz for robot with feed type we want
    "robot_on": "RC1/PLC8-Infeed_Robot/HMI/Feed_Robot_Mode/Value".lower(),
    "rpm": "RC1/4420-Calciner/4420-KLN-001_RPM/Input".lower(),
    "kiln_weight": "RC1/4420-Calciner/4420-WT-0007_Alarm/Input".lower(),
    "large_mod_feed": "RC1/4420-Calciner/Module_Loading/HMI/Cycle_Time_Complete/Value".lower(),
    "n2_cons": "RC1/4420-Calciner/4420-FT-0001_Vol_Flow_Ave/Input".lower(),
    "zone_1_temp": "RC1/4420-Calciner/4420-TE-7210-B_AI/Input_Scaled".lower(),
    "zone_2_temp": "RC1/4420-Calciner/4420-TE-7220-B_AI/Input_Scaled".lower(),
    "zone_3_temp": "RC1/4420-Calciner/4420-TE-7230-B_AI/Input_Scaled".lower(),
    "load_cell_1": "RC1/4420-Calciner/4420-WT-0007/Status/S_RealTimeWeight".lower(),
    "load_cell_3": "RC1/4420-Calciner/4420-WT-0009/Status/S_RealTimeWeight".lower(),
    "robot_on": "RC1/PLC8-Infeed_Robot/HMI/Feed_Robot_Mode/Value".lower(),
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
        # zero_start = pd.Timestamp("2025-07-17 04:30", tz="UTC")
        # zero_end = pd.Timestamp("2025-07-22 07:30", tz="UTC")
        # df = loadcell_dif_zeroer(df, zero_start, zero_end)

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

            # zero_start = pd.Timestamp("2025-07-17 04:30", tz="UTC")
            # zero_end = pd.Timestamp("2025-07-22 07:30", tz="UTC")
            # df = loadcell_dif_zeroer(df, zero_start, zero_end)
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
    rpm = df["rpm"].to_numpy()
    angle = 0  # degrees

    t_avg = []
    wt_avg = []
    rpm_avg = []
    motor_amps_avg = []
    small_mod_feed_avg = []
    large_mod_feed_avg = []
    n2_cons_avg = []
    zone1_temp_avg = []
    zone2_temp_avg = []
    zone3_temp_avg = []
    load_cell_1_avg = []
    load_cell_3_avg = []
    robot_on = []
    small_mod_sp = []
    load_cell_1_rev = 0
    load_cell_3_rev = 0

    revs_complete = 0
    wt_rev = 0
    rpm_rev = 0
    motor_amps_rev = 0
    small_mod_feed_rev = 0
    robot_on_rev = 0
    large_mod_feed_rev = 0
    n2_cons_rev = 0
    zone1_temp_rev = 0
    zone2_temp_rev = 0
    zone3_temp_rev = 0
    num_pts_rev = 0

    for i in range(1, len(rpm)):
        angle += rpm[i] * (t[i] - t[i - 1]) / np.timedelta64(1, "s") / 60 * 360

        wt_rev += df.iloc[i, df.columns.get_loc("kiln_weight")]
        rpm_rev += rpm[i]
        motor_amps_rev += df.iloc[i, df.columns.get_loc("motor_amps")]
        zone1_temp_rev += df.iloc[i, df.columns.get_loc("zone_1_temp")]
        # Add zone 2 and 3 temp
        zone2_temp_rev += df.iloc[i, df.columns.get_loc("zone_2_temp")]
        zone3_temp_rev += df.iloc[i, df.columns.get_loc("zone_3_temp")]
        small_mod_feed_rev += df.iloc[i, df.columns.get_loc("small_mod_feed")]
        robot_on_rev += df.iloc[i, df.columns.get_loc("robot_on")]
        large_mod_feed_rev += df.iloc[i, df.columns.get_loc("large_mod_feed")]
        n2_cons_rev += df.iloc[i, df.columns.get_loc("n2_cons")]
        # Add load cell diff
        load_cell_1_rev += df.iloc[i, df.columns.get_loc("load_cell_1")]
        load_cell_3_rev += df.iloc[i, df.columns.get_loc("load_cell_3")]

        num_pts_rev += 1

        if angle >= 360 * (revs_complete + 1):
            revs_complete += 1
            t_avg.append(t[i])
            rpm_avg.append(rpm_rev / num_pts_rev)
            wt_avg.append(wt_rev / num_pts_rev)
            motor_amps_avg.append(motor_amps_rev / num_pts_rev)
            small_mod_feed_avg.append(small_mod_feed_rev / num_pts_rev)
            large_mod_feed_avg.append(large_mod_feed_rev / num_pts_rev)
            n2_cons_avg.append(n2_cons_rev / num_pts_rev)
            zone1_temp_avg.append(zone1_temp_rev / num_pts_rev)
            zone2_temp_avg.append(zone2_temp_rev / num_pts_rev)
            zone3_temp_avg.append(zone3_temp_rev / num_pts_rev)
            load_cell_1_avg.append(load_cell_1_rev / num_pts_rev)
            load_cell_3_avg.append(load_cell_3_rev / num_pts_rev)
            match = df.loc[df["timestamp"] == t[i], "robot_on"]
            robot_on.append(match.values[0] if not match.empty else np.nan)
            small_mod_sp.append(df.loc[df["timestamp"] == t[i], "small_mod_feed"].values[0])

            wt_rev = 0
            rpm_rev = 0
            motor_amps_rev = 0
            small_mod_feed_rev = 0
            large_mod_feed_rev = 0
            n2_cons_rev = 0
            zone1_temp_rev = 0
            zone2_temp_rev = 0
            zone3_temp_rev = 0
            load_cell_1_rev = 0
            load_cell_3_rev = 0
            num_pts_rev = 0

    df_avg = pd.DataFrame(
        {
            "timestamp": np.array(t_avg),
            "kiln_weight": np.array(wt_avg),
            "motor_amps": np.array(motor_amps_avg),
            "rpm": np.array(rpm_avg),
            "zone_1_temp": np.array(zone1_temp_avg),
            "zone_2_temp": np.array(zone2_temp_avg),
            "zone_3_temp": np.array(zone3_temp_avg),
            "small_mod_feed": np.array(small_mod_feed_avg),
            "robot_on": np.array(robot_on),
            "large_mod_feed": np.array(large_mod_feed_avg),
            "n2_cons": np.array(n2_cons_avg),
            "loadcell_diff": np.array(load_cell_1_avg) - np.array(load_cell_3_avg),
            "load_cell_1": np.array(load_cell_1_avg),
            "load_cell_3": np.array(load_cell_3_avg),
            "small_mod_sp": np.array(small_mod_sp),
        }
    )

    return df_avg


def loadcell_dif_zeroer(df, start_time=None, end_time=None, loadcell_col="load_cell_1"):
    """
    Zero the load cell difference values in the DataFrame.
    If start_time and end_time are provided, only zero the values in that range.
    The function subtracts the initial value in the range from all values in the range.
    """
    df = df.copy()
    if start_time is not None and end_time is not None:
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        if mask.any():
            initial_value = abs(df.loc[mask, loadcell_col].iloc[0])
            print(initial_value)
            df.loc[mask, loadcell_col] = df.loc[mask, loadcell_col] - initial_value
            df["loadcell_diff"] = df["load_cell_1"] - df["load_cell_3"]

    return df


def run():
    start_utc = "2025-06-15 00:00:00"
    stop_utc = "2025-08-04 00:00:00"

    df, df_avg = load_data(start_utc, stop_utc)

    # Add a label column to distinguish the two DataFrames
    df_labeled = df.copy()
    df_labeled["source"] = "raw"
    df_avg_labeled = df_avg.copy()
    df_avg_labeled["source"] = "avg"
    print(df_avg_labeled.head())

    # Align columns for concatenation
    all_cols = sorted(set(df_labeled.columns) | set(df_avg_labeled.columns))
    df_labeled = df_labeled.reindex(columns=all_cols)
    df_avg_labeled = df_avg_labeled.reindex(columns=all_cols)

    mask = ((df_avg["rpm"] < 0.08) & (df_avg["kiln_weight"] < 700)) | (df_avg["motor_amps"] < 0.1)
    df_avg = df_avg[~mask]
    df_avg = loadcell_dif_zeroer(df_avg)

    df_avg = loadcell_dif_zeroer(
        df_avg,
        start_time=pd.Timestamp("2025-07-17 04:30", tz="UTC"),
        end_time=pd.Timestamp("2025-07-22 07:30", tz="UTC"),
    )
    df_avg["loadcell_diff"] = abs(df_avg["loadcell_diff"])

    # Write df_avg to the Current Data folder
    output_dir = "Current Spike/Current Data"
    os.makedirs(output_dir, exist_ok=True)
    avg_out_path = os.path.join(output_dir, "updated_avg_motor_current.csv")
    df_avg.to_csv(avg_out_path, index=False)
    print(f"Wrote averaged data to {avg_out_path}")

    plt.plot(df_avg["timestamp"], df_avg["motor_amps"], label="Motor Amps (Avg)", color="blue")
    plt.xlabel("Timestamp")
    plt.ylabel("Motor Amps (Avg)")
    plt.title("Motor Amps (Avg) Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
