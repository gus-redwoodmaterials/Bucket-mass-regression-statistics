import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

CLEANOUTS = pd.to_datetime(
    [
        "2025-06-02 7:00",
        "2025-06-03 7:00",
        "2025-06-22 7:00",
        "2025-06-24 7:00",
        "2025-06-25 7:00",  # Wed
        "2025-07-03 7:00",
        "2025-07-08 7:00",
        "2025-07-12 7:00",
        "2025-07-15 7:00",
        "2025-07-21 7:00",
        "2025-07-23 7:00",  # Wed
        "2025-07-25 7:00",
        "2025-07-29 7:00",
    ],
    utc=True,  # same as tz_localize("UTC") after parse
)

# All Wednesdays in the range, at **7:00 UTC**
WEDNESDAYS = pd.date_range(start="2025-06-04", end="2025-08-04", freq="W-WED", tz="UTC") + pd.Timedelta(hours=7)

# Union (unique + sorted) instead of concat
CLEANOUTS = CLEANOUTS.union(WEDNESDAYS).sort_values()


def final_presentation_plotter(df, start_dt, end_dt, show_third_ax=False, custom_title=None, bat_name="312"):
    bat_columns = [col for col in df.columns if bat_name in col]
    if not bat_columns:
        print(f"No Bat {bat_name} columns found in analysis_df.")
        return
    bat_name = bat_columns[0]
    # yaxis_upper = df[bat_name].max() * 1.05
    # yaxis_lower = df[bat_name].min() * 0.95
    helper = bat_name.split("___", 1)[1] if "___" in bat_name else bat_name
    if helper == "ultium_pouch":
        helper = "Ultium 312"
    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
    df_masked = df.loc[mask]
    if show_third_ax:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(
        df_masked["timestamp"],
        df_masked[bat_name],
        label=f"{custom_title} {helper}",
        color="tab:blue",
    )
    polymer_laptop = [col for col in df.columns if "polymer_laptop" in col]
    if True:
        ax1.plot(
            df_masked["timestamp"],
            df_masked[polymer_laptop].mean(axis=1),
            label=f"{custom_title} Polymer Laptop",
            color="tab:red",
        )
    # ax1.set_ylim(yaxis_lower, yaxis_upper)
    # Add vertical lines for each cleanout
    if hasattr(CLEANOUTS, "__iter__") and not isinstance(CLEANOUTS, (str, bytes, pd.Timestamp)):
        cleanout_list = list(CLEANOUTS)
    else:
        cleanout_list = [CLEANOUTS]
    for i, cleanout_time in enumerate(cleanout_list):
        if cleanout_time >= df_masked["timestamp"].min() and cleanout_time <= df_masked["timestamp"].max():
            ax1.axvline(cleanout_time, color="k", linestyle="--", alpha=0.7, label="Cleanout" if i == 0 else None)
    ax1.set_ylabel(f"Total Batteries", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax2.plot(df_masked["timestamp"], df_masked["residual_stage1"], label="Amps - fit(rpm,wt)", color="tab:orange")
    for i, cleanout_time in enumerate(cleanout_list):
        if cleanout_time >= df_masked["timestamp"].min() and cleanout_time <= df_masked["timestamp"].max():
            ax2.axvline(cleanout_time, color="k", linestyle="--", alpha=0.7, label="Cleanout" if i == 0 else None)
    ax2.set_ylabel("Residuals", fontsize=14)
    ax2.set_xlabel("Timestamp", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True)
    if show_third_ax:
        ax3a = ax3
        ax3b = ax3.twinx()
        l1 = ax3a.plot(df_masked["timestamp"], df_masked["rpm"], label="RPM", color="tab:red")
        l2 = ax3b.plot(df_masked["timestamp"], df_masked["kiln_weight"], label="Kiln Weight", color="tab:green")
        ax3a.set_ylabel("RPM", fontsize=14, color="tab:red")
        ax3b.set_ylabel("Kiln Weight", fontsize=14, color="tab:green")
        ax3a.tick_params(axis="y", labelcolor="tab:red")
        ax3b.tick_params(axis="y", labelcolor="tab:green")
        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        ax3a.legend(lines, labels, loc="upper right")
        ax3a.grid(True)
    # Calculate hours range for title
    hours_range = (end_dt - start_dt).total_seconds() / 3600.0
    plt.suptitle(
        f"Increase in Total Ultium 312s -> Increase in Average Amps",
        fontsize=18,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.show()


def plot_bat_vs_residuals(df, start_dt, end_dt, show_third_ax=False, custom_title=None, bat_name="312"):
    bat_columns = [col for col in df.columns if bat_name in col]
    if not bat_columns:
        print(f"No Bat {bat_name} columns found in analysis_df.")
        return
    bat_name = bat_columns[0]
    yaxis_upper = df[bat_name].max() * 1.05
    yaxis_lower = df[bat_name].min() * 0.95
    helper = bat_name.split("___", 1)[1] if "___" in bat_name else bat_name
    if helper == "ultium_pouch":
        helper = "Ultium 312"
    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
    df_masked = df.loc[mask]
    if show_third_ax:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(
        df_masked["timestamp"],
        df_masked[bat_name],
        label=f"{custom_title} {helper}",
        color="tab:blue",
    )
    ax1.set_ylim(yaxis_lower, yaxis_upper)
    # Add vertical lines for each cleanout
    if hasattr(CLEANOUTS, "__iter__") and not isinstance(CLEANOUTS, (str, bytes, pd.Timestamp)):
        cleanout_list = list(CLEANOUTS)
    else:
        cleanout_list = [CLEANOUTS]
    for i, cleanout_time in enumerate(cleanout_list):
        if cleanout_time >= df_masked["timestamp"].min() and cleanout_time <= df_masked["timestamp"].max():
            ax1.axvline(cleanout_time, color="k", linestyle="--", alpha=0.7, label="Cleanout" if i == 0 else None)
    ax1.set_ylabel(f"{custom_title if custom_title else ''} {helper}", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax2.plot(df_masked["timestamp"], df_masked["residual_stage1"], label="Amps - fit(rpm,wt)", color="tab:orange")
    for i, cleanout_time in enumerate(cleanout_list):
        if cleanout_time >= df_masked["timestamp"].min() and cleanout_time <= df_masked["timestamp"].max():
            ax2.axvline(cleanout_time, color="k", linestyle="--", alpha=0.7, label="Cleanout" if i == 0 else None)
    ax2.set_ylabel("Amps - fit(rpm,wt)", fontsize=14)
    ax2.set_xlabel("Timestamp", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True)
    if show_third_ax:
        ax3a = ax3
        ax3b = ax3.twinx()
        l1 = ax3a.plot(df_masked["timestamp"], df_masked["rpm"], label="RPM", color="tab:red")
        l2 = ax3b.plot(df_masked["timestamp"], df_masked["kiln_weight"], label="Kiln Weight", color="tab:green")
        ax3a.set_ylabel("RPM", fontsize=14, color="tab:red")
        ax3b.set_ylabel("Kiln Weight", fontsize=14, color="tab:green")
        ax3a.tick_params(axis="y", labelcolor="tab:red")
        ax3b.tick_params(axis="y", labelcolor="tab:green")
        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        ax3a.legend(lines, labels, loc="upper right")
        ax3a.grid(True)
    # Calculate hours range for title
    hours_range = (end_dt - start_dt).total_seconds() / 3600.0
    plt.suptitle(
        f"Increase in average Amps - controlling for RPM",
        fontsize=18,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.show()


def plot_loadcell_diff(df, start_dt, end_dt, custom_title=None):
    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
    df_masked = df.loc[mask]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    # Top plot: loadcell_diff
    ax1.plot(
        df_masked["timestamp"],
        df_masked["loadcell_diff"],
        label="Loadcell Diff",
        color="tab:purple",
    )
    # Add vertical lines for each cleanout
    if hasattr(CLEANOUTS, "__iter__") and not isinstance(CLEANOUTS, (str, bytes, pd.Timestamp)):
        cleanout_list = list(CLEANOUTS)
    else:
        cleanout_list = [CLEANOUTS]
    for i, cleanout_time in enumerate(cleanout_list):
        if cleanout_time >= df_masked["timestamp"].min() and cleanout_time <= df_masked["timestamp"].max():
            ax1.axvline(cleanout_time, color="k", linestyle="--", alpha=0.7, label="Cleanout" if i == 0 else None)
    ax1.set_ylabel("Loadcell Diff", fontsize=14)
    if custom_title:
        title = f"{custom_title} Loadcell Diff\n{start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M}"
    else:
        title = f"Loadcell Diff\n{start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M}"
    ax1.set_title(title, fontsize=16)
    ax1.legend(loc="upper right")
    ax1.grid(True)
    # Bottom plot: motor_amps
    ax2.plot(
        df_masked["timestamp"],
        df_masked["motor_amps"],
        label="Motor Amps",
        color="tab:blue",
    )
    for i, cleanout_time in enumerate(cleanout_list):
        if cleanout_time >= df_masked["timestamp"].min() and cleanout_time <= df_masked["timestamp"].max():
            ax2.axvline(cleanout_time, color="k", linestyle="--", alpha=0.7, label="Cleanout" if i == 0 else None)
    ax2.set_ylabel("Motor Amps", fontsize=14)
    ax2.set_xlabel("Timestamp", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


def run():
    analysis_cumulative_df = pd.read_csv("Current Spike/results/scout_analysis_df_cumulative_standardized.csv")
    analysis_lookback_df = pd.read_csv("Current Spike/results/scout_analysis_df_lookback_60min_standardized.csv")

    analysis_cumulative_df["timestamp"] = pd.to_datetime(analysis_cumulative_df["timestamp"], utc=True)
    analysis_lookback_df["timestamp"] = pd.to_datetime(analysis_lookback_df["timestamp"], utc=True)
    # Default: plot full range
    start_dt = analysis_cumulative_df["timestamp"].min()
    end_dt = analysis_cumulative_df["timestamp"].max()
    series_2_start = pd.Timestamp("2025-07-12 01:00", tz="UTC")
    series_2_end = pd.Timestamp("2025-07-12 04:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df,
    #     start_dt,
    #     end_dt,
    #     show_third_ax=False,
    #     custom_title="Cumulative",
    #     bat_name="059",
    # )
    # plot_bat_vs_residuals(
    #     analysis_lookback_df,
    #     series_2_start,
    #     series_2_end,
    #     show_third_ax=False,
    #     custom_title="Lookback",
    #     bat_name="312",
    # )
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df,
    #     series_2_start,
    #     series_2_end,
    #     show_third_ax=False,
    #     custom_title="Cumulative",
    #     bat_name="059",
    # )
    final_presentation_plotter(
        analysis_lookback_df,
        series_2_start,
        series_2_end,
        show_third_ax=False,
        custom_title="Feedrate of",
        bat_name="312",
    )
    series_2_start = pd.Timestamp("2025-07-02 07:00", tz="UTC")
    series_2_end = pd.Timestamp("2025-07-03 07:00", tz="UTC")
    final_presentation_plotter(
        analysis_cumulative_df,
        series_2_start,
        series_2_end,
        show_third_ax=False,
        custom_title="Cumulative",
        bat_name="312",
    )
    # start_1 = pd.Timestamp("2025-07-10 00:00", tz="UTC")
    # end_1 = pd.Timestamp("2025-07-13 00:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df, start_1, end_1, show_third_ax=False, custom_title="Cumulative", bat_name="784"
    # )
    # start_2 = pd.Timestamp("2025-07-16 00:00", tz="UTC")
    # end_2 = pd.Timestamp("2025-07-20 00:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df, start_2, end_2, show_third_ax=False, custom_title="Cumulative", bat_name="784"
    # )
    # start_3 = pd.Timestamp("2025-07-22 00:00", tz="UTC")
    # end_3 = pd.Timestamp("2025-07-23 00:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_lookback_df, start_3, end_3, show_third_ax=False, custom_title="Feedrate", bat_name="784"
    # )
    # plot_bat_vs_residuals(analysis_cumulative_df, start_dt, end_dt, show_third_ax=False, custom_title="Cumulative")
    # series_1_start = pd.Timestamp("2025-06-19 00:00", tz="UTC")
    # series_1_end = pd.Timestamp("2025-06-23 00:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df, series_1_start, series_1_end, show_third_ax=False, custom_title="Cumulative"
    # )
    # series_2_start = pd.Timestamp("2025-07-02 07:00", tz="UTC")
    # series_2_end = pd.Timestamp("2025-07-03 07:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df, series_2_start, series_2_end, show_third_ax=False, custom_title="Cumulative"
    # )
    # series_3_start = pd.Timestamp("2025-07-17 12:00", tz="UTC")
    # series_3_end = pd.Timestamp("2025-07-18 16:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_cumulative_df, series_3_start, series_3_end, show_third_ax=False, custom_title="Cumulative"
    # )
    # look_back_start1 = pd.Timestamp("2025-06-22 00:00", tz="UTC")
    # look_back_end1 = pd.Timestamp("2025-06-22 03:00", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_lookback_df, look_back_start1, look_back_end1, show_third_ax=False, custom_title="Feedrate of"
    # )
    # look_back_start2 = pd.Timestamp("2025-07-12 01:00", tz="UTC")
    # look_back_end2 = pd.Timestamp("2025-07-12 04:00", tz="UTC")
    # look_back_start3 = pd.Timestamp("2025-06-21 05:45", tz="UTC")
    # look_back_end3 = pd.Timestamp("2025-06-21 08:45", tz="UTC")
    # plot_bat_vs_residuals(
    #     analysis_lookback_df, look_back_start2, look_back_end2, show_third_ax=False, custom_title="Feedrate of"
    # )
    # plot_bat_vs_residuals(
    #     analysis_lookback_df, look_back_start3, look_back_end3, show_third_ax=False, custom_title="Feedrate of"
    # )
    # plot_bat_vs_residuals(
    #     analysis_lookback_df, start_dt, end_dt, show_third_ax=False, custom_title="Feedrate of All Data", bat_nam="058"
    # )
    # start = pd.Timestamp("2025-07-17 04:30", tz="UTC")
    # end = pd.Timestamp("2025-07-22 07:30", tz="UTC")
    # plot_loadcell_diff(analysis_lookback_df, start_dt, end_dt, custom_title="Loadcell Diff of All Data")


if __name__ == "__main__":
    run()
