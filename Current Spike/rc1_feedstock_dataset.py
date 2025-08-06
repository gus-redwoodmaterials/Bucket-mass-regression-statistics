# ------------------------------------------------------------
# CSV DEPENDENCIES (Queried from AWS Athena):
#
# This script automatically queries and downloads the following files
# from AWS Athena if they are not already present in the local directory.
#
# Files:
#   1. scout_ingest_raw_061525_072225.csv
#   2. scout_backfilled_raw.csv  ← Make sure to specify the correct model version
#   3. infeed_cvr_hz_061525_072225.csv
# ------------------------------------------------------------


import pandas as pd
import numpy as np
import os
import time
import boto3
import yaml


with open("Current Spike/config.yaml", "r") as f:
    config = yaml.safe_load(f)


INGEST_ID_CSV_PATH = config["csv_paths"]["ingest_ids"]
SCOUT_CSV_PATH = config["csv_paths"]["scout_predictions"]
INFEED_CVR_HZ_CSV_PATH = config["csv_paths"]["infeed_cvr_hz"]
OUTPUT_CSV_PATH = config["outputs"]["scout_buckets_dataset"]

QUERY_START_TIMESTAMP_UTC = config["query_range"]["start_utc"]
QUERY_STOP_TIMESTAMP_UTC = config["query_range"]["stop_utc"]
MODEL_VERSION = config["scout_model_version"]

# t_entering_kiln_utc global constants
BUCKETS_TILL_END = config["conveyor"]["buckets_till_end"]
AVG_BUCKET_LENGTH = config["conveyor"]["avg_bucket_length_m"]
CVR_LENGTH_REM = BUCKETS_TILL_END * AVG_BUCKET_LENGTH  # 16.95 m


def query_csv(output_csv_path: str, query: str, database: str, workgroup: str = "primary") -> None:
    """
    Queries AWS Athena and downloads the resulting CSV file to the local directory
    if the file does not already exist.
    """
    if os.path.exists(output_csv_path):
        print(f"Found: {output_csv_path}")
        return

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


def get_csvs() -> None:
    """
    Ensures that all required CSV files for the pipeline are present in the local directory.
    """
    query_csv(
        output_csv_path=INGEST_ID_CSV_PATH,
        query=f"""
        SELECT *
        FROM "raw"."rw_battery_detector_ingest"
        WHERE timestamp >= TIMESTAMP '{QUERY_START_TIMESTAMP_UTC}'
          AND "timestamp" < TIMESTAMP '{QUERY_STOP_TIMESTAMP_UTC}'
        ORDER BY timestamp ASC;
        """,
        database="raw",
    )

    query_csv(
        output_csv_path=SCOUT_CSV_PATH,
        query=f"""
        SELECT *
        FROM "raw"."rw_battery_detector_predictions"
        WHERE model_version = '{MODEL_VERSION}'
        ORDER BY timestamp ASC;
        """,
        database="raw",
    )

    query_csv(
        output_csv_path=INFEED_CVR_HZ_CSV_PATH,
        query=f"""
        SELECT tagpath,
               t_stamp,
               float_value,
               "timestamp"
        FROM "cleansed"."rc1_historian"
        WHERE "timestamp" >= TIMESTAMP '{QUERY_START_TIMESTAMP_UTC}'
          AND "timestamp" <= TIMESTAMP '{QUERY_STOP_TIMESTAMP_UTC}'
          AND tagpath = '{config["tagpaths"]["infeed_cvr_hz"]}'
        ORDER BY "timestamp";
        """,
        database="cleansed",
    )


def load_ingest_id_times(input_csv: str):
    """
    Matches each ingest_id with its start timestamp.
    """
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "timestamp_utc", "id": "ingest_id"})

    ingest_id_times = (
        df.groupby("ingest_id")["timestamp_utc"]
        .min()
        .dt.round("1s")
        .reset_index(name="timestamp_utc")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return ingest_id_times


def get_scout_predictions(input_csv: str, ingest_id_times: pd.DataFrame):
    """
    Matches each scout ingest_id & prediction with its ingest timestamp
    """
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    # Only keep predictions where ingest_id exists in ingest_id_times
    df = df[df["ingest_id"].isin(ingest_id_times["ingest_id"])]

    # Clean prediction names by removing number prefixes and colons
    df["prediction_clean"] = (
        df["prediction"].str.split(":").str[-1].str.strip().replace({"Jumpstarter": "R1000419 - Noco Jumpstarter"})
    )

    # Pivots so each row has an ingest_id and its corresponding predictions
    pred_counts_pivot = df.groupby(["ingest_id", "prediction_clean"]).size().unstack(fill_value=0).reset_index()

    pred_counts_df = pd.merge(ingest_id_times, pred_counts_pivot, on="ingest_id", how="left")
    pred_counts_df = pred_counts_df.sort_values("timestamp_utc").reset_index(drop=True)

    prediction_cols = [
        col
        for col in pred_counts_df.columns
        if col
        not in [
            "ingest_id",
            "timestamp_utc",
        ]
    ]

    # Drop rows where all prediction counts are NaN or zero
    valid_rows = ~((pred_counts_df[prediction_cols].fillna(0) == 0).all(axis=1))
    valid_scout_df = pred_counts_df[valid_rows].sort_values("timestamp_utc").reset_index(drop=True)

    return valid_scout_df


def load_infeed_hz_csv(raw_csv_path: str) -> pd.DataFrame:
    """
    Loads and formats infeed conveyor Hz data, computes speed and distance per second.
    """
    df = pd.read_csv(raw_csv_path)
    df = df.rename(columns={"timestamp": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    column_renames = {"rc1_4420-calciner_4420-cvr-001_status_speed_feedback_hz": "infeed_cvr_hz"}

    formatted_df = format_and_interpolate(df, column_renames)
    formatted_df["infeed_cvr_hz"] = formatted_df["infeed_cvr_hz"].round(2)

    # Computes seconds to move 1 bucket distance (1.13m) with a given Hz
    formatted_df["sec_per_bucket"] = formatted_df["infeed_cvr_hz"].apply(seconds_per_bucket).round(1)

    # Computes the distance per second
    formatted_df["dist_per_s"] = (1.13 / formatted_df["sec_per_bucket"]).round(5)
    formatted_df = formatted_df.sort_values("timestamp_utc").reset_index(drop=True)

    return formatted_df.set_index("timestamp_utc")


def seconds_per_bucket(hz: float) -> float:
    """
    Given an infeed conveyor speed (Hz), returns seconds per bucket.
    """

    HZ_CONVEY_CURVE = [0, 0.5, 1, 2, 4, 6, 9, 12, 15, 17, 20]
    SEC_PER_BUCKET_CURVE = [1e9, 300, 140, 60, 29, 21, 13, 9, 7, 6, 5]

    sec = np.interp(hz, HZ_CONVEY_CURVE, SEC_PER_BUCKET_CURVE)
    return sec + 20


def format_and_interpolate(df: pd.DataFrame, column_renames: dict, resample_interval: str = "1s") -> pd.DataFrame:
    """
    Pivots, renames, resamples, and interpolates time-series sensor data.
    """
    # Pivots df so each tagpath becomes a column. The mean is taken in case of duplicate timestamps.
    df = df.pivot_table(
        index=["timestamp_utc"],
        columns="tagpath",
        values="float_value",
        aggfunc="mean",
    ).reset_index()

    df.columns = [col.replace("/", "_") if isinstance(col, str) else col for col in df.columns]
    df = df.rename(columns=column_renames)

    # Resamples to 1-second intervals using mean, then resets index.
    df = df.set_index("timestamp_utc").resample(resample_interval).mean().reset_index()

    def custom_interp(s: pd.Series) -> pd.Series:
        s = s.copy()
        idx = s.dropna().index
        if idx.empty:
            return s
        s.iloc[: idx[0]] = s.iloc[idx[0]]  # Fill leading NaNs with first valid value
        s.iloc[idx[-1] + 1 :] = s.iloc[idx[-1]]  # Fill trailing NaNs with last valid value
        return s.ffill()

    df["infeed_cvr_hz"] = custom_interp(df["infeed_cvr_hz"])
    return df


def compute_t_entering_kiln(
    start_ts: pd.Timestamp, infeed_cvr_hz_df: pd.DataFrame, conveyor_length_m: float
) -> pd.Timestamp:
    """
    Computes estimated time when a bucket reaches the end of the conveyor.
    """
    end_ts = start_ts + pd.Timedelta(hours=1)
    window = infeed_cvr_hz_df.loc[start_ts:end_ts]
    cum_dist = window["dist_per_s"].cumsum()

    # Find the index where cumulative distance first exceeds the conveyor length
    idx = np.searchsorted(cum_dist.values, conveyor_length_m, side="left")

    if idx < len(cum_dist):  # if idx = len(cum_dist) means conveyor length not reached
        return cum_dist.index[idx]
    else:
        fallback_window = infeed_cvr_hz_df.loc[start_ts:]
        fallback_cum_dist = fallback_window["dist_per_s"].cumsum()
        fallback_idx = np.searchsorted(fallback_cum_dist.values, conveyor_length_m, side="left")
        return fallback_cum_dist.index[fallback_idx] if fallback_idx < len(fallback_cum_dist) else pd.NaT


def add_t_entering_kiln_column(
    scout_preds_df: pd.DataFrame,
    infeed_cvr_hz_df: pd.DataFrame,
    conveyor_length_m: float,
) -> pd.DataFrame:
    """
    Adds t_entering_kiln_utc and travel_time_min columns to scout_preds_df and returns selected columns.
    """
    scout_preds_df["t_entering_kiln_utc"] = scout_preds_df["timestamp_utc"].apply(
        lambda ts: compute_t_entering_kiln(ts, infeed_cvr_hz_df, conveyor_length_m)
    )

    scout_preds_df["travel_time_min"] = (
        (scout_preds_df["t_entering_kiln_utc"] - scout_preds_df["timestamp_utc"]).dt.total_seconds().div(60).round(2)
    )
    scout_preds_df = scout_preds_df.sort_values("t_entering_kiln_utc").reset_index(drop=True)
    # Insert the new columns after the second column
    cols = list(scout_preds_df.columns)
    scout_preds_df = scout_preds_df[
        cols[:2]
        + ["t_entering_kiln_utc", "travel_time_min"]
        + [c for c in cols[2:] if c not in ("t_entering_kiln_utc", "travel_time_min")]
    ]

    return scout_preds_df


def add_category_counts(
    bucket_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    battery_col: str = "predicted_class",  # column in mapping_df
    category_col: str = "engineering_categorization",  # column in mapping_df
) -> pd.DataFrame:
    """
    For each row (bucket) in `bucket_df`, sum the counts of batteries
    that belong to the same `category_col` and append those sums as
    new columns.

    Parameters
    ----------
    bucket_df : pd.DataFrame
        Wide table: one column per battery type, numeric counts.
    mapping_df : pd.DataFrame
        Long table: each row links a battery type to a category.
        Must contain columns `battery_col` and `category_col`.
    battery_col : str, default "Predicted Class"
    category_col : str, default "Category"

    Returns
    -------
    pd.DataFrame
        Copy of `bucket_df` with extra columns named
        f"cnt_{category}" for every distinct category.
    """

    # --- 1. Build a mapping {battery_type: category} --------------------------
    battery_to_cat = mapping_df.set_index(battery_col)[category_col]

    # Keep only battery columns that actually exist in bucket_df
    valid_batteries = [b for b in battery_to_cat.index if b in bucket_df.columns]

    if not valid_batteries:
        raise ValueError("None of the battery types in mapping_df match the columns in bucket_df.")

    # --- 2. Group battery columns by category ---------------------------------
    cat_to_batts = (
        battery_to_cat.loc[valid_batteries]
        .groupby(level=0)  # groupby category
        .groups  # {category: Index([...battery types...])}
    )

    # --- 3. Sum counts row-wise for each category -----------------------------
    out = bucket_df.copy()

    for cat, cols in cat_to_batts.items():
        out[f"cnt_{cat}"] = bucket_df[cols].sum(axis=1).astype(int)

    return out


def main():
    get_csvs()
    ingest_id_times = load_ingest_id_times(INGEST_ID_CSV_PATH)
    scout_preds_df = get_scout_predictions(SCOUT_CSV_PATH, ingest_id_times)
    infeed_cvr_hz_df = load_infeed_hz_csv(INFEED_CVR_HZ_CSV_PATH)
    scout_offset_df = add_t_entering_kiln_column(scout_preds_df, infeed_cvr_hz_df, CVR_LENGTH_REM)

    scout_offset_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
