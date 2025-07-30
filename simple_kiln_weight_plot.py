#!/usr/bin/env python3
"""
Simple script to plot kiln weight for May 5th, 2-3pm California time
This is for debugging data alignment issues
"""

import pandas as pd
import matplotlib.pyplot as plt
import pytz
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download

# Configuration
TABLE_PATH = "cleansed.rc1_historian"
DATA_TIMESTEP_SECONDS = 3
pacific_tz = pytz.timezone("US/Pacific")

# Only grab the kiln weight tag we need
TAGS = {
    "kiln_weight_avg": "rc1/4420-calciner/hmi/kilnweightavg/value",
    "bucket_mass": "rc1/4420-calciner/acmestatus/bucket_mass",
}


def main():
    print("🔍 Simple Kiln Weight Plot - May 5th, 2-3pm California Time")

    # Define exact time range: May 5th, 2025, 2:00 PM to 3:00 PM Pacific
    start_str = "2025-05-05T14:00:00"  # 2:00 PM
    end_str = "2025-05-05T15:00:00"  # 3:00 PM

    # Parse and localize to Pacific timezone
    start_time = pacific_tz.localize(parse(start_str))
    end_time = pacific_tz.localize(parse(end_str))

    print(f"Fetching data from {start_time} to {end_time}")
    print(f"Timezone: {start_time.tzinfo}")

    # Fetch data from Athena
    try:
        df = athena_download.get_pivoted_athena_data(
            TAGS,
            start_time,
            end_time,
            TABLE_PATH,
            DATA_TIMESTEP_SECONDS,
        )

        print(f"✅ Data loaded: {len(df)} rows")

        if len(df) == 0:
            print("❌ No data returned!")
            return

        # Print first few timestamps to verify
        print(f"\nFirst 5 timestamps:")
        for i in range(min(5, len(df))):
            print(f"  {df.iloc[i]['timestamp']}")

        print(f"\nLast 5 timestamps:")
        for i in range(max(0, len(df) - 5), len(df)):
            print(f"  {df.iloc[i]['timestamp']}")

        # Basic data info
        print(f"\nData summary:")
        print(f"  Total points: {len(df)}")
        print(f"  Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")

        if "kiln_weight_avg" in df.columns:
            valid_weights = df["kiln_weight_avg"].dropna()
            print(f"  Valid weight points: {len(valid_weights)}")
            print(f"  Weight range: {valid_weights.min():.1f} to {valid_weights.max():.1f} kg")
            print(f"  Average weight: {valid_weights.mean():.1f} kg")
        else:
            print("  ❌ No kiln_weight_avg column found!")
            print(f"  Available columns: {list(df.columns)}")
            return

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["kiln_weight_avg"], "b-", linewidth=1, alpha=0.7)
        plt.xlabel("Time (Pacific)")
        plt.ylabel("Kiln Weight (kg)")
        plt.title("Kiln Weight - May 5, 2025, 2:00-3:00 PM Pacific Time")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Save data to CSV for inspection
        csv_filename = "kiln_weight_may5_2to3pm.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 Data saved to {csv_filename}")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
