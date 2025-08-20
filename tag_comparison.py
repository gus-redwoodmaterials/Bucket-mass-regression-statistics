#!/usr/bin/env python3
"""
Tag Comparison Script

This script reads a JSON file containing tag data with timestamps and values,
compares two specified tags at matching timestamps, and outputs the results to a CSV file.

Usage:
    python tag_comparison.py <json_file_path> <tag1> <tag2> [output_csv]

Example:
    python tag_comparison.py data.json "Tag.COOLING_DAMPER_1" "Tag.HEATING_DAMPER_1" output.csv
"""

import json
import pandas as pd
import sys
from datetime import datetime


def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        sys.exit(1)


def compare_tags(data, tag1, tag2):
    """
    Compare two tags and return a DataFrame with matching timestamps.

    Args:
        data (dict): JSON data loaded from file
        tag1 (str): First tag name
        tag2 (str): Second tag name

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'tag1_name', 'tag1_value', 'tag2_name', 'tag2_value']
    """
    # Check if both tags exist in the data
    if tag1 not in data:
        print(f"Error: Tag '{tag1}' not found in the data.")
        available_tags = list(data.keys())[:10]  # Show first 10 tags
        print(f"Available tags (first 10): {available_tags}")
        sys.exit(1)

    if tag2 not in data:
        print(f"Error: Tag '{tag2}' not found in the data.")
        available_tags = list(data.keys())[:10]  # Show first 10 tags
        print(f"Available tags (first 10): {available_tags}")
        sys.exit(1)

    tag1_data = data[tag1]
    tag2_data = data[tag2]

    # Find common timestamps
    common_timestamps = set(tag1_data.keys()) & set(tag2_data.keys())

    if not common_timestamps:
        print(f"Error: No common timestamps found between '{tag1}' and '{tag2}'.")
        sys.exit(1)

    # Create comparison data
    comparison_data = []
    for timestamp in sorted(common_timestamps):
        comparison_data.append(
            {
                "timestamp": timestamp,
                "tag1_name": tag1,
                "tag1_value": tag1_data[timestamp],
                "tag2_name": tag2,
                "tag2_value": tag2_data[timestamp],
            }
        )

    df = pd.DataFrame(comparison_data)

    # Convert timestamp to datetime for better sorting and display
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def generate_output_filename(tag1, tag2):
    """Generate a default output filename based on tag names and current time."""
    # Clean tag names for filename (remove special characters)
    clean_tag1 = "".join(c for c in tag1 if c.isalnum() or c in ("-", "_")).rstrip()
    clean_tag2 = "".join(c for c in tag2 if c.isalnum() or c in ("-", "_")).rstrip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tag_comparison_{clean_tag1}_vs_{clean_tag2}_{timestamp}.csv"


def print_summary(df, tag1, tag2):
    """Print a summary of the comparison."""
    print("\n=== Tag Comparison Summary ===")
    print(f"Tag 1: {tag1}")
    print(f"Tag 2: {tag2}")
    print(f"Total matching timestamps: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if len(df) > 0:
        print("\nTag 1 statistics:")
        print(f"  Mean: {df['tag1_value'].mean():.4f}")
        print(f"  Min: {df['tag1_value'].min():.4f}")
        print(f"  Max: {df['tag1_value'].max():.4f}")

        print("\nTag 2 statistics:")
        print(f"  Mean: {df['tag2_value'].mean():.4f}")
        print(f"  Min: {df['tag2_value'].min():.4f}")
        print(f"  Max: {df['tag2_value'].max():.4f}")


def main():
    # Specify your file path and tags here
    json_file = "/Users/gus.robinson/Desktop/Local Github Repos/rw-acme-to/docker_image/artifacts/2025_08_20_16_42_33/test_controllers.py/test_offline[mar_12_25]/new_writes.json"
    tag1 = "Tag.TO_O2_OUTPUT"
    tag2 = "Tag.FRESH_AIR_VALVE_PERCENT"  # Change to your desired second tag
    output_csv = None  # Set to a filename if you want a custom output, else leave as None

    print(f"Loading data from '{json_file}'...")
    data = load_json_data(json_file)
    print(f"Loaded data for {len(data)} tags.")

    print(f"Comparing tags '{tag1}' and '{tag2}'...")
    df = compare_tags(data, tag1, tag2)

    if output_csv:
        output_file = output_csv
    else:
        output_file = generate_output_filename(tag1, tag2)

    df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'")

    print("\nFirst 5 rows of comparison data:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
