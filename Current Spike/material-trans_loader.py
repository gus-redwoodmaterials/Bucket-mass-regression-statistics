from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import os
import sys
import boto3
import time

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from dateutil.parser import parse

# Configuration
DATA_FOLDER = "data"

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Using the working tag names from your SQL query
TAGS = {
    "module_manual_load_conveyor_lane_a": "Module Manual Load Conveyor Lane A",
    "manual_loading": "RC1 Infeed Manual Loading",
    "infeed_tipper_a": "RC1 Infeed Tipper A",
    "infeed_tipper_b": "RC1 Infeed Tipper B",
    "robot": "RC1 OEM Robot Loader A",
}

# Database configuration
TABLE_PATH = "raw.incoming_material_transaction_v"
pacific_tz = pytz.timezone("US/Pacific")
DATA_TIMESTEP_SECONDS = 3
SMOOTH_SIDE_POINTS = 5  # ±5 → 11-point window


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
    csv_filename = f"material_trans_{description}_{start_str}_to_{end_str}.csv"
    csv_path = os.path.join(DATA_FOLDER, csv_filename)

    print(f"\n📊 Loading {description} data: {start.strftime('%m/%d %H:%M')} to {end.strftime('%m/%d %H:%M')}")

    # Check command line arguments for refresh flag
    refresh_data = "--refresh" in sys.argv or "-r" in sys.argv

    # Try to load from cache first
    if not refresh_data and os.path.exists(csv_path):
        print("   Loading from cache...")
        df = pd.read_csv(csv_path)
        # Convert timestamp back to datetime
        df["transaction_time_utc"] = pd.to_datetime(df["transaction_time_utc"])
    else:
        print("   Fetching from Athena...")
        # Convert timezone-aware datetime to UTC string for SQL query
        start_utc = start.astimezone(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
        end_utc = end.astimezone(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Build SQL query matching your working SQL
        location_descriptions = "', '".join(TAGS.values())
        sql_query = f"""
        SELECT *
        FROM {TABLE_PATH}
        WHERE location_id LIKE 'RC1%'
          AND location_description IN ('{location_descriptions}')
          AND transaction_time_utc >= TIMESTAMP '{start_utc}'
          AND transaction_time_utc < TIMESTAMP '{end_utc}'
          AND error_flag IS NULL
        ORDER BY transaction_time_utc DESC
        """

        # Use the new query_csv function
        query_csv(csv_path, sql_query, "raw")
        df = pd.read_csv(csv_path)
        df["transaction_time_utc"] = pd.to_datetime(df["transaction_time_utc"], utc=True)
        print("   Cached for future use")

    # Ensure timestamp is in Pacific timezone
    return df


def run():
    query_csv(
        output_csv_path=os.path.join(DATA_FOLDER, "mat_trans.csv"),
        query="""
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
                and transaction_time_utc >= TIMESTAMP '2025-06-15 00:00:00'
                and transaction_time_utc < TIMESTAMP '2025-07-22 00:00:00'
                and error_flag is NULL
                order by transaction_time desc;
        """,
        database="raw",
    )


if __name__ == "__main__":
    run()
