from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import os
import sys
import boto3
import time
import amp_feed_coorelation

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# Configuration
DATA_FOLDER = "data"

CLEANOUTS = pd.to_datetime(
    [
        "2025-06-02 12:00",
        "2025-06-03 12:00",
        "2025-06-22 12:00",
        "2025-06-24 12:00",
        "2025-06-25 12:00",
        "2025-07-03 12:00",
        "2025-07-08 12:00",
        "2025-07-12 12:00",
        "2025-07-15 12:00",
        "2025-07-21 12:00",
        "2025-07-23 12:00",
        "2025-07-25 12:00",
        "2025-07-29 12:00",
    ]
).tz_localize("utc")


def count_batteries_per_window(
    df: pd.DataFrame,
    start_time,  # str | pd.Timestamp | datetime
    window_size: int = 60,  # minutes
    time_col: str = "transaction_time_utc",
):
    """
    Sum battery counts over a window of `window_size` minutes that starts at
    `start_time`, using `time_col` as the timestamp for alignment.

    Returns
    -------
    dict
        {battery_type_column: total_count_in_window, ...}
        (Columns with a zero sum are dropped.)
    """

    # 2.  Define the window
    start = pd.to_datetime(start_time, utc=True)
    end = start + pd.Timedelta(minutes=window_size)

    mask = (df.index >= start) & (df.index < end)

    # 3.  Identify battery-type columns (everything after the travel_time column)
    non_battery_cols = {"ingest_id", "timestamp_utc", "transaction_time_utc", "travel_time_min"}
    battery_cols = [c for c in df.columns if c not in non_battery_cols]

    # 4.  Sum counts in the window
    counts = df.loc[mask, battery_cols].sum()

    # 5.  Return only non-zero entries as a plain dict
    return counts


def query_csv(output_csv_path: str, query: str, database: str, workgroup: str = "primary") -> pd.DataFrame:
    """
    Queries AWS Athena and downloads the resulting CSV file to the local directory
    if the file does not already exist.
    """
    if os.path.exists(output_csv_path):
        print(f"Found: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    s3_bucket = "athena-results.redwoodmaterials.us-west-2"
    s3_prefix = "primary/"

    s3_output = f"s3://{s3_bucket}/{s3_prefix}"

    print(f"{output_csv_path} not found. Querying Athena...")
    athena = boto3.client("athena")
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": s3_output},
        WorkGroup=workgroup,
    )
    query_execution_id = response["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=query_execution_id)["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(2)

    if status != "SUCCEEDED":
        raise Exception(f"Athena query failed: {status}")

    s3 = boto3.client("s3")
    key = f"{s3_prefix}{query_execution_id}.csv"
    s3.download_file(s3_bucket, key, output_csv_path)
    print(f"Downloaded: {output_csv_path}")
    return pd.read_csv(output_csv_path)


def make_buckets_df(materials_df, offset_hours=1):
    materials_df["transaction_time_utc"] = pd.to_datetime(materials_df["transaction_time_utc"])
    battery_cols = materials_df["item_number"].unique().tolist()
    rows = []
    for _, row in materials_df.iterrows():
        transaction_time = pd.to_datetime(row["transaction_time_utc"] - pd.Timedelta(hours=offset_hours), utc=True)
        row_dict = {col: 0 for col in battery_cols}
        row_dict[row["item_number"]] = row["quantity"]
        row_dict["transaction_time_utc"] = transaction_time
        rows.append(row_dict)
    df = pd.DataFrame(rows)
    return df


def create_analysis_dataframe(amps_df, buckets_df, battery_cols, start_time, sample_interval="1T", window_size=60):
    """
    Create a dataframe that combines motor amps data with battery counts in a rolling window.

    Parameters
    ----------
    amps_df : pd.DataFrame
        DataFrame containing motor amps data. Must have a 'timestamp_utc' column.
    buckets_df : pd.DataFrame
        DataFrame containing battery data. Must have a 'transaction_time_utc' column.
    battery_cols : list
        List of column names in buckets_df that represent battery types.
    start_time : pd.Timestamp or str
        The start time for the analysis. Only data after this time will be processed.
    sample_interval : str, default "1T"
        The interval for resampling the data. Uses pandas offset aliases (e.g., "1T" for 1 minute).
    window_size : int, default 60
        The size of the rolling window in minutes for counting batteries.
        If set to -1, uses a cumulative sum of all batteries up to each timestamp
        instead of a rolling window.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing timestamps, motor amps, and battery counts for each time point.
    """
    # Set the timestamp column as the index
    amps_df = amps_df.set_index("timestamp")

    # Sample at regular intervals to reduce computation time
    sampled_timestamps = amps_df[amps_df.index >= start_time].resample(sample_interval).first().dropna()
    print(sampled_timestamps.head())
    print(f"Analyzing {len(sampled_timestamps)} time points...")

    # Create a new DataFrame for analysis
    rows = []
    last_cleanout = CLEANOUTS[CLEANOUTS <= start_time].max()
    # For each timestamp, get motor amps and battery counts in previous hour
    for i, timestamp in enumerate(sampled_timestamps.index):
        print(f"Processing point {i + 1}/{len(sampled_timestamps)}...")

        # Get motor amps at this timestamp (amps_df is indexed by timestamp_utc)
        if timestamp in amps_df.index:
            val = amps_df.loc[timestamp, "motor_amps"]
            # If multiple rows, take the mean (or first)
            if isinstance(val, pd.Series):
                motor_amps_val = val.mean()
            else:
                motor_amps_val = val
            temp_1_val = amps_df.loc[timestamp, "zone_1_temp"] if "zone_1_temp" in amps_df.columns else np.nan
            temp_2_val = amps_df.loc[timestamp, "zone_2_temp"] if "zone_2_temp" in amps_df.columns else np.nan
            temp_3_val = amps_df.loc[timestamp, "zone_3_temp"] if "zone_3_temp" in amps_df.columns else np.nan
            rpm_val = amps_df.loc[timestamp, "rpm"] if "rpm" in amps_df.columns else np.nan
            kiln_weight_val = amps_df.loc[timestamp, "kiln_weight"] if "kiln_weight" in amps_df.columns else np.nan
            loadcell_diff_val = (
                amps_df.loc[timestamp, "loadcell_diff"] if "loadcell_diff" in amps_df.columns else np.nan
            )
        else:
            # Find closest timestamp if exact match not found
            closest_idx = np.argmin(np.abs(amps_df.index - timestamp))
            motor_amps_val = amps_df.iloc[closest_idx]["motor_amps"]
            temp_1_val = amps_df.iloc[closest_idx]["zone_1_temp"] if "zone_1_temp" in amps_df.columns else np.nan
            temp_2_val = amps_df.iloc[closest_idx]["zone_2_temp"] if "zone_2_temp" in amps_df.columns else np.nan
            temp_3_val = amps_df.iloc[closest_idx]["zone_3_temp"] if "zone_3_temp" in amps_df.columns else np.nan
            rpm_val = amps_df.iloc[closest_idx]["rpm"] if "rpm" in amps_df.columns else np.nan
            kiln_weight_val = amps_df.iloc[closest_idx]["kiln_weight"] if "kiln_weight" in amps_df.columns else np.nan
            loadcell_diff_val = (
                amps_df.iloc[closest_idx]["loadcell_diff"] if "loadcell_diff" in amps_df.columns else np.nan
            )

        # Get battery counts and cleanout info
        if window_size == -1:
            last_cleanout = CLEANOUTS[CLEANOUTS <= timestamp].max()
            # Cumulative sum: use all batteries from beginning up to this timestamp
            mask = (buckets_df.index <= timestamp) & (buckets_df.index > last_cleanout)
            window_counts = buckets_df.loc[mask, battery_cols].sum()
            # Calculate time since last cleanout
            time_since_cleanout = (timestamp - last_cleanout).total_seconds() / 60.0  # in minutes
            # Calculate total number of batteries since last cleanout
            total_batteries_since_cleanout = buckets_df.loc[mask, battery_cols].sum().sum()
        else:
            # Rolling window: use batteries in previous hour
            window_start = timestamp - pd.Timedelta(hours=1)
            window_counts = count_batteries_per_window(
                buckets_df,
                start_time=window_start,
                window_size=window_size,  # Default 60 minutes = 1 hour
                time_col="transaction_time_utc",
            )
            # For rolling window, set time since cleanout and total batteries since cleanout to NaN
            time_since_cleanout = np.nan
            total_batteries_since_cleanout = np.nan

        # Create row with timestamp, motor amps, battery counts, and new columns
        row = {
            "timestamp": timestamp,
            "motor_amps": motor_amps_val,
            "zone_1_temp": temp_1_val,
            "zone_2_temp": temp_2_val,
            "zone_3_temp": temp_3_val,
            "rpm": rpm_val,
            "kiln_weight": kiln_weight_val,
            "loadcell_diff": loadcell_diff_val,
            "time_since_cleanout": time_since_cleanout,
            "total_batteries_since_cleanout": total_batteries_since_cleanout,
        }

        # Add battery counts (0 for batteries not in the window)
        for col in battery_cols:
            row[col] = window_counts.get(col, 0)

        rows.append(row)

    # Create DataFrame with all data
    df = pd.DataFrame(rows)
    return df


def run():
    # query_csv(
    #     output_csv_path=os.path.join(DATA_FOLDER, "mat_trans.csv"),
    #     query="""
    #             select *
    #             from incoming_material_transaction_v
    #             where location_id like 'RC1%'
    #             and location_description in (
    #                 'Module Manual Load Conveyor Lane A',
    #                 'RC1 Infeed Manual Loading',
    #                 'RC1 Infeed Tipper A',
    #                 'RC1 Infeed Tipper B',
    #                 'RC1 OEM Robot Loader A'
    #             )
    #             and transaction_time_utc >= TIMESTAMP '2025-06-15 00:00:00'
    #             and transaction_time_utc < TIMESTAMP '2025-07-22 00:00:00'
    #             and error_flag is NULL
    #             order by transaction_time desc;
    #     """,
    #     database="raw",
    # )

    start_utc = "2025-06-15 00:00:00"
    stop_utc = "2025-08-04 00:00:00"
    material_df = query_csv(
        output_csv_path=os.path.join(DATA_FOLDER, f"material_trans_{start_utc}_{stop_utc}.csv"),
        query=f"""
                select *
                from incoming_material_transaction_v
                where location_id like 'RC1%'
                and location_description in (
                    'Module Manual Load Conveyor Lane A',
                    'RC1 Infeed Manual Loading',
                    'RC1 Infeed Tipper A',
                    'RC1 Infeed Tipper B',
                    'RC1 OEM Robot Loader A'
                )
                and transaction_time_utc >= TIMESTAMP '{start_utc}'
                and transaction_time_utc < TIMESTAMP '{stop_utc}'
                and error_flag is NULL
                order by transaction_time desc;
        """,
        database="raw",
    )

    amps_df = pd.read_csv("Current Spike/Current Data/updated_avg_motor_current.csv")
    buckets_df = make_buckets_df(material_df)
    buckets_df["transaction_time_utc"] = pd.to_datetime(buckets_df["transaction_time_utc"])
    amps_df["timestamp"] = pd.to_datetime(amps_df["timestamp"])

    battery_cols = material_df["item_number"].unique().tolist()
    print(battery_cols)
    print(buckets_df.columns)
    print(amps_df.head())
    start_time = max(amps_df["timestamp"].min(), buckets_df["transaction_time_utc"].min()) + pd.Timedelta(hours=1)

    buckets_df.columns = (
        buckets_df.columns.str.replace("`", "", regex=False)
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.lower()
    )
    amps_df.columns = (
        amps_df.columns.str.replace("`", "", regex=False)
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.lower()
    )

    # Update battery_cols to match the cleaned column names
    battery_cols = [
        col.replace("`", "").replace(" ", "_").replace("-", "_").replace("/", "_").lower() for col in battery_cols
    ]

    # Set the timestamp as index for buckets_df
    buckets_df = buckets_df.set_index("transaction_time_utc")

    # Define analysis type for naming
    window_size = -1
    analysis_df = create_analysis_dataframe(
        amps_df, buckets_df, battery_cols, start_time, sample_interval="1T", window_size=window_size
    )

    if len(analysis_df) > 0:
        print(f"\nCreated dataset with {len(analysis_df)} rows and {len(analysis_df.columns)} columns")
        investigation_cols = battery_cols.copy()
        # investigation_cols = category_cols
        # Calculate correlations between motor amps and each battery type
        correlations, sorted_correlations = amp_feed_coorelation.calculate_category_correlations(
            analysis_df, investigation_cols
        )

        # Determine analysis type for filename
        analysis_type = "cumulative" if window_size == -1 else "lookback"
        results_folder = "Current Spike/results"
        os.makedirs(results_folder, exist_ok=True)

        # Define predictors for regression
        # Only include time_since_cleanout and total_batteries_since_cleanout if they exist and are not all NaN
        base_predictors = ["rpm", "zone_1_temp", "zone_2_temp", "zone_3_temp", "kiln_weight", "loadcell_diff"]
        if "time_since_cleanout" in analysis_df.columns and not analysis_df["time_since_cleanout"].isna().all():
            base_predictors.append("time_since_cleanout")
        if (
            "total_batteries_since_cleanout" in analysis_df.columns
            and not analysis_df["total_batteries_since_cleanout"].isna().all()
        ):
            base_predictors.append("total_batteries_since_cleanout")

        # Run multiple linear regression
        # CHANGE ME!
        STANDARDIZE = True  # Ensure we are standardizing predictors
        model_std, beta_tbl = amp_feed_coorelation.run_standardised_regression(
            analysis_df=analysis_df,
            base_predictors=base_predictors,
            category_predictors=investigation_cols,
            print_results=True,
        )

        # Add standardized flag to filename if applicable
        standardized_suffix = "_standardized" if STANDARDIZE else ""

        # Write standardised regression coefficients to CSV
        regression_filename = (
            f"{results_folder}/material_trans_{analysis_type}{standardized_suffix}_regression_impacts.csv"
        )
        # Write index as 'var' column
        beta_tbl_out = beta_tbl.reset_index().rename(columns={beta_tbl.index.name or 'index': 'var'})
        beta_tbl_out.to_csv(regression_filename, index=False)
        print(f"Standardised regression results written to {regression_filename}")


if __name__ == "__main__":
    run()
