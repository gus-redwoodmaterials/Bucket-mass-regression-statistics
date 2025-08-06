import pandas as pd


def add_category_counts(
    bucket_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    battery_col: str = "Battery Type",  # column in mapping_df
    category_col: str = "predicted_class",  # column in mapping_df
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
    battery_col : str, default "Battery Type"
    category_col : str, default "Category"

    Returns
    -------
    pd.DataFrame
        Copy of `bucket_df` with extra columns named
        f"cnt_{category}" for every distinct category.
    """

    out = bucket_df.copy()
    # List of unique categories from your examples
    unique_categories = ["cat_plastic", "cat_mixed", "cat_mod", "cat_steel", "cat_scrap", "cat_pouch"]
    # Add a column for each category, initialized to 0
    for cat in unique_categories:
        out[cat] = 0

    non_battery_cols = {"ingest_id", "timestamp_utc", "t_entering_kiln_utc", "travel_time_min"}
    battery_cols = [c for c in bucket_df.columns if c not in non_battery_cols]
    battery_to_categories = {}

    for _, row in mapping_df.iterrows():
        battery = row[battery_col]
        # Split categories by comma, strip whitespace, and filter out empty strings
        categories = [cat.strip() for cat in str(row[category_col]).split(",") if cat.strip()]
        battery_to_categories[battery] = categories

    for _, row in bucket_df.iterrows():
        battery_counts = row[battery_cols]
        for battery, categories in battery_to_categories.items():
            if battery in battery_counts:
                count = battery_counts[battery]
                if count is None or pd.isna(count):
                    count = 0
                for category in categories:
                    out.loc[row.name, f"cat_{category}"] = out.loc[row.name, f"cat_{category}"] + count

    return out


def run():
    # Load data
    buckets_df = pd.read_csv("Current Spike/Current Data/scout_buckets_dataset.csv")
    amps_df = pd.read_csv("Current Spike/Current Data/avg_motor_current.csv")
    buckets_df.columns = buckets_df.columns.str.replace(" ", "_").str.replace("-", "_").str.replace("/", "_")
    category_df = pd.read_csv("Current Spike/Current Data/scout_categories.csv")

    category_df["predicted_class"] = (
        category_df["predicted_class"].str.replace(" ", "_").str.replace("-", "_").str.replace("/", "_")
    )

    print(buckets_df.columns)
    print(category_df["predicted_class"])

    buckets_df = add_category_counts(
        bucket_df=buckets_df,
        mapping_df=category_df,
        battery_col="predicted_class",
        category_col="engineering_categorization",
    )
    buckets_df.to_csv("Current Spike/Current Data/scout_buckets_dataset_with_categories.csv", index=False)


if __name__ == "__main__":
    run()
