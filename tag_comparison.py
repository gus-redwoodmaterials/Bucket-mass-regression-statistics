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
import matplotlib.pyplot as plt


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


def compare_tags(data, tags):
    """
    Compare multiple tags and return a DataFrame with matching timestamps.

    Args:
        data (dict): JSON data loaded from file
        tags (list): List of tag names to compare

    Returns:
        pd.DataFrame: DataFrame with timestamp and value columns for each tag
    """
    # Check which tags exist in the data and get their data
    tag_data = {}
    for tag in tags:
        if tag not in data:
            print(f"Warning: Tag '{tag}' not found in data")
        else:
            tag_data[tag] = data[tag]

    if not tag_data:
        print("Error: No valid tags found in data")
        sys.exit(1)

    # Get timestamps for each tag
    tag_timestamps = {tag: set(tag_data[tag].keys()) for tag in tag_data}

    # Find common timestamps across all tags
    common_timestamps = set.intersection(*tag_timestamps.values())

    if not common_timestamps:
        print("Error: No common timestamps found across all tags")
        sys.exit(1)

    # Create comparison data
    comparison_data = []
    for timestamp in sorted(common_timestamps):
        row_data = {"timestamp": timestamp}
        for tag in tag_data:
            row_data[f"{tag}_value"] = tag_data[tag][timestamp]
        comparison_data.append(row_data)

    df = pd.DataFrame(comparison_data)

    # Convert timestamp to datetime for better sorting and display
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def main():
    # Specify your file path and tags here
    json_file = "/Users/gus.robinson/Desktop/Local Github Repos/rw-acme-to/docker_image/artifacts/2025_09_05_20_21_47/test_controllers.py/test_offline[mar_12_25]/new_writes.json"

    output_csv = "Tag_comparison.csv"  # Set to a filename if you want a custom output, else leave as None

    tags = ["Tag.FEED_RATE_REQUESTED", "Tag.FEED_RATE_TARGET", "Tag.HZ_VAC_FAN"]

    print(f"Loading data from '{json_file}'...")
    data = load_json_data(json_file)
    print(f"Loaded data for {len(data)} tags.")

    print(f"Comparing tags '{tags[0]}' and '{tags[1]}'...")
    df = compare_tags(data, tags)

    df.to_csv(output_csv, index=False)
    print(f"Results saved to '{output_csv}'")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    # First tag on top subplot
    axes[0].plot(df["timestamp"], df[f"{tags[0]}_value"], color="blue")
    axes[0].set_ylabel(f"Actual Feed Rate")
    axes[0].set_title(f"Actual Feed Rate Over Time")
    axes[0].grid(True)

    # Second tag on bottom subplot
    axes[1].plot(df["timestamp"], df[f"{tags[1]}_value"], color="orange")
    axes[1].set_ylabel(f"Margin Limited Feed Rate")
    axes[1].set_title(f"Margin Limited Feed Rate Over Time")
    axes[1].grid(True)

    axes[1].set_xlabel("Timestamp")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nFirst 5 rows of comparison data:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
