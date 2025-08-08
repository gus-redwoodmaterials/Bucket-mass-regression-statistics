import pandas as pd
import numpy as np
import pytz
import os
import sys
from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler

STANDARDIZE = False  # Set to True to use cumulative battery counts instead of rolling window

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
    time_col: str = "t_entering_kiln_utc",
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
    # 1.  Make sure the timestamp column is datetime-typed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()  # avoid mutating caller’s df
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # 2.  Define the window
    start = pd.to_datetime(start_time, utc=True)
    end = start + pd.Timedelta(minutes=window_size)

    mask = (df[time_col] >= start) & (df[time_col] < end)

    # 3.  Identify battery-type columns (everything after the travel_time column)
    non_battery_cols = {"ingest_id", "timestamp_utc", "t_entering_kiln_utc", "travel_time_min"}
    battery_cols = [c for c in df.columns if c not in non_battery_cols]

    # 4.  Sum counts in the window
    counts = df.loc[mask, battery_cols].sum()

    # 5.  Return only non-zero entries as a plain dict
    return counts


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


def add_impact_table(model, df, prefix="impact_", q_lo=0.10, q_hi=0.90):
    """
    Compute 10-90 %ile 'impact' for every slope in a statsmodels OLS/GLS result.

    Returns
    -------
    DataFrame with columns:
        coef, pvalue, q10, q90, impact, impact_abs
    """
    coefs = model.params
    pvals = model.pvalues

    q10 = df.quantile(q_lo)
    q90 = df.quantile(q_hi)
    span = q90 - q10

    out = pd.DataFrame({"coef": coefs, "pvalue": pvals, "q10": q10, "q90": q90, "impact": coefs * span})
    out["impact_abs"] = out["impact"].abs()
    # drop the intercept (no span)
    return out.drop(index="Intercept", errors="ignore").sort_values("impact_abs", ascending=False)


def calculate_category_correlations(analysis_df, category_cols, min_data_points=5):
    """
    Calculate correlations between motor amps and each battery category.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        DataFrame containing motor amps and battery/category counts.
    category_cols : list
        List of column names in analysis_df that represent battery categories.
    min_data_points : int, default 5
        Minimum number of data points required to calculate correlation.

    Returns
    -------
    dict
        Dictionary of correlation statistics for each category.
    list
        Sorted list of (category, stats) tuples by absolute correlation.
    """
    # Calculate correlations for categories
    correlations = {}
    for col in category_cols:
        # Only include rows where this category was present
        df_filtered = analysis_df[analysis_df[col] > 0]
        if len(df_filtered) > min_data_points:  # Require at least min_data_points data points
            corr = df_filtered["motor_amps"].corr(df_filtered[col])
            correlations[col] = {
                "correlation": corr,
                "data_points": len(df_filtered),
                "avg_count": df_filtered[col].mean(),
                "avg_amps": df_filtered["motor_amps"].mean(),
            }

    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)

    return correlations, sorted_correlations


def run_multiple_linear_regression(analysis_df, base_predictors, category_predictors=None, print_results=True):
    """
    Run multiple linear regression analysis on the provided dataset.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        DataFrame containing the data for analysis.
    base_predictors : list
        List of base predictor column names (e.g., ["rpm", "zone_1_temp", "kiln_weight"]).
    category_predictors : list, optional
        List of category predictor column names. If None, will use columns starting with "cat_".
    print_results : bool, default True
        Whether to print the regression results.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted regression model.
    pd.DataFrame
        Table of coefficient impacts.
    """
    # If category_predictors is not provided, use all columns starting with "cat_"
    if category_predictors is None:
        category_predictors = [c for c in analysis_df.columns if c.startswith("cat_")]

    all_predictors = base_predictors + category_predictors

    if print_results:
        print("\n=== MULTIPLE-LINEAR REGRESSION ===")
        print(f"Modelling motor_amps as a function of {', '.join(base_predictors)}, and battery-category counts")
        print(f"Including {len(all_predictors)} predictors")

    # Use sanitized column names directly in formula
    formula = "motor_amps ~ " + " + ".join(all_predictors)

    if print_results:
        print(f"OLS formula: {formula}")

    # Ensure data column names are strings
    analysis_df.columns = analysis_df.columns.astype(str)

    # Fit the ordinary-least-squares model
    try:
        ols_model = ols(formula, data=analysis_df).fit(cov_type="HC3")  # HC3 = robust SE
    except Exception as e:
        print(f"Error during regression: {str(e)}")
        print("Column names in dataframe:", analysis_df.columns.tolist())
        raise

    if print_results:
        # Print a tidy summary table
        print("\n--- Coefficients (robust SE) ---")

        # Try common alternatives for the p-value column
        cols = ols_model.summary2().tables[1].columns
        pval_col = next((c for c in cols if "P>|" in c or "P>|t" in c or "P>|z" in c or "P-value" in c), None)
        coef_cols = ["Coef.", "Std.Err.", pval_col, "[0.025", "0.975]"]
        coef_df = (
            ols_model.summary2()
            .tables[1]
            .loc[:, [c for c in coef_cols if c in cols]]
            .rename(
                columns={
                    "Coef.": "estimate",
                    "Std.Err.": "std_err",
                    pval_col: "p_value",
                    "[0.025": "ci_low",
                    "0.975]": "ci_high",
                }
            )
        )
        print(coef_df.to_string(float_format=lambda x: f"{x:8.3f}"))

        # Highlight statistically-significant predictors (p < 0.05)
        sig = coef_df[coef_df["p_value"] < 0.05]
        if len(sig):
            print("\nSignificant predictors (p < 0.05):")
            for idx, row in sig.iterrows():
                print(
                    f"  {idx:<25}:  {row['estimate']:8.3f}  A  "
                    f"(95 % CI {row['ci_low']:.3f} – {row['ci_high']:.3f}, "
                    f"p = {row['p_value']:.4f})"
                )
        else:
            print("\nNo predictors reached p < 0.05 in this window.")

        # Overall model diagnostics
        print("\n--- Model fit ---")
        print(f"  R²        : {ols_model.rsquared:.3f}")
        print(f"  Adj. R²   : {ols_model.rsquared_adj:.3f}")
        print(f"  Obs       : {int(ols_model.nobs)}")
        print(f"  F-statistic (robust) p-value: {ols_model.f_pvalue:.4g}")

    impact_tbl = add_impact_table(ols_model, analysis_df[all_predictors])

    if print_results:
        print("\n=== 10-90 %ile impact (amps) ===")
        print(impact_tbl[["coef", "impact", "pvalue"]].head(10))

    return ols_model, impact_tbl


def create_analysis_dataframe(amps_df, buckets_df, battery_cols, start_time, sample_interval="1T", window_size=60):
    """
    Create a dataframe that combines motor amps data with battery counts in a rolling window.

    Parameters
    ----------
    amps_df : pd.DataFrame
        DataFrame containing motor amps data. Must have a 'timestamp_utc' column.
    buckets_df : pd.DataFrame
        DataFrame containing battery data. Must have a 't_entering_kiln_utc' column.
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
    amps_df = amps_df.set_index("timestamp_utc")

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
            rpm_val = amps_df.loc[timestamp, "rpm"] if "rpm" in amps_df.columns else np.nan
            kiln_weight_val = amps_df.loc[timestamp, "kiln_weight"] if "kiln_weight" in amps_df.columns else np.nan
            loadcell_diff_val = (
                amps_df.loc[timestamp, "loadcell_diff"] if "loadcell_diff" in amps_df.columns else np.nan
            )
            temp_2_val = amps_df.loc[timestamp, "zone_2_temp"] if "zone_2_temp" in amps_df.columns else np.nan
            temp_3_val = amps_df.loc[timestamp, "zone_3_temp"] if "zone_3_temp" in amps_df.columns else np.nan
        else:
            # Find closest timestamp if exact match not found
            closest_idx = np.argmin(np.abs(amps_df.index - timestamp))
            motor_amps_val = amps_df.iloc[closest_idx]["motor_amps"]
            temp_1_val = amps_df.iloc[closest_idx]["zone_1_temp"] if "zone_1_temp" in amps_df.columns else np.nan
            rpm_val = amps_df.iloc[closest_idx]["rpm"] if "rpm" in amps_df.columns else np.nan
            kiln_weight_val = amps_df.iloc[closest_idx]["kiln_weight"] if "kiln_weight" in amps_df.columns else np.nan
            loadcell_diff_val = (
                amps_df.iloc[closest_idx]["loadcell_diff"] if "loadcell_diff" in amps_df.columns else np.nan
            )
            temp_2_val = amps_df.iloc[closest_idx]["zone_2_temp"] if "zone_2_temp" in amps_df.columns else np.nan
            temp_3_val = amps_df.iloc[closest_idx]["zone_3_temp"] if "zone_3_temp" in amps_df.columns else np.nan

        # Get battery counts and cleanout info
        if window_size == -1:
            last_cleanout = CLEANOUTS[CLEANOUTS <= timestamp].max()
            # Cumulative sum: use all batteries from beginning up to this timestamp
            mask = (buckets_df["t_entering_kiln_utc"] <= timestamp) & (
                buckets_df["t_entering_kiln_utc"] > last_cleanout
            )
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
                time_col="t_entering_kiln_utc",
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


# ---------------------------------------------------------------------------
#  Standardised-regression helper
# ---------------------------------------------------------------------------


def run_standardised_regression(
    analysis_df: pd.DataFrame,
    base_predictors: list,
    category_predictors: list | None = None,
    print_results: bool = True,
):
    """
    Fit an OLS model after z-scaling every predictor (mean-0, std-1). The
    absolute value of each resulting coefficient is directly comparable: it
    represents the change in motor_amps produced by a *one-standard-deviation*
    move in that predictor.

    Returns
    -------
    model_std  : statsmodels RegressionResultsWrapper
    beta_table : pd.DataFrame   # tidy view of standardised betas
    """
    STANDARDIZE = True  # Ensure we are standardizing predictors
    if category_predictors is None:
        category_predictors = [c for c in analysis_df.columns if c.startswith("cat_")]

    X_cols = base_predictors + category_predictors
    y = analysis_df["motor_amps"]

    # ---- 1.  Standard-scale X ------------------------------------------------
    scaler = StandardScaler()
    X_z = pd.DataFrame(
        scaler.fit_transform(analysis_df[X_cols]),
        columns=X_cols,
        index=analysis_df.index,
    )
    X_z = sm.add_constant(X_z)

    # ---- 2.  Fit OLS with robust (HC3) SEs ----------------------------------
    model_std = sm.OLS(y, X_z).fit(cov_type="HC3")

    # ---- 3.  Tidy summary of standardised betas -----------------------------
    # Get p-values for each predictor (excluding intercept)
    pvalues = model_std.pvalues.drop("const")
    beta = (
        model_std.params.drop("const")  # we don’t rank the intercept
        .to_frame("beta_std")
        .assign(abs_beta=lambda d: d["beta_std"].abs(), p_value=pvalues)
        .sort_values("abs_beta", ascending=False)
    )

    if print_results:
        print("\n=== STANDARDISED (Z-SCORE) REGRESSION ===")
        print(f"Included predictors: {', '.join(X_cols)}")
        print("\nTop 10 predictors by |β| (amps per 1 SD move):")
        print(beta.head(10).to_string(float_format=lambda x: f"{x:8.3f}"))
        print(
            f"\nModel R²: {model_std.rsquared:.3f}   Adj R²: {model_std.rsquared_adj:.3f}   n = {int(model_std.nobs)}"
        )

    return model_std, beta


def run():
    # Load data
    buckets_df = pd.read_csv("Current Spike/Current Data/scout_buckets_dataset_with_categories.csv")
    amps_df = pd.read_csv("Current Spike/Current Data/updated_avg_motor_current.csv")

    # After loading the data, sanitize column names
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
    # Ensure timestamps are in datetime format
    buckets_df["t_entering_kiln_utc"] = pd.to_datetime(buckets_df["t_entering_kiln_utc"], utc=True)
    amps_df["timestamp_utc"] = pd.to_datetime(amps_df["timestamp"], utc=True)

    # Find common time range
    start_time = max(amps_df["timestamp_utc"].min(), buckets_df["t_entering_kiln_utc"].min()) + pd.Timedelta(hours=1)

    print(f"Processing data from {start_time}...")

    # Identify all battery type columns
    category_cols = [c for c in buckets_df.columns if c.startswith("cat_")]
    non_battery_cols = {"ingest_id", "timestamp_utc", "t_entering_kiln_utc", "travel_time_min"}
    battery_cols = [c for c in buckets_df.columns if c not in non_battery_cols]
    # Create safe column names with underscores

    # Use the create_analysis_dataframe function to generate analysis data
    window_size = -1  # Use rolling window analysis
    STANDARDIZE = True
    analysis_df = create_analysis_dataframe(
        amps_df=amps_df,
        buckets_df=buckets_df,
        battery_cols=battery_cols,  # Pass the underscore versions
        start_time=start_time,
        sample_interval="1T",  # 1 minute interval - adjust as needed
        window_size=window_size,  # Use rolling window
    )
    analysis_df.columns = (
        analysis_df.columns.str.replace("`", "", regex=False)
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.lower()
    )
    # mask = (analysis_df["rpm"] < 0.08) & (analysis_df["kiln_weight"] < 700)
    # analysis_df = analysis_df[~mask]  # Remove rows where rpm < 0.08 and kiln_weight < 700

    if len(analysis_df) > 0:
        print(f"\nCreated dataset with {len(analysis_df)} rows and {len(analysis_df.columns)} columns")
        investigation_cols = [col for col in battery_cols if col not in category_cols]
        # investigation_cols = category_cols
        # Calculate correlations between motor amps and each battery type
        correlations, sorted_correlations = calculate_category_correlations(analysis_df, investigation_cols)

        # --- Plot: Feed rate of battery type containing '312' and motor_amps as time series (masked 7/1 to 7/5) ---

        # Find the battery column containing '312'
        # battery_col_312 = next((col for col in analysis_df.columns if "312" in col), None)
        # if battery_col_312:
        #     bat_name = battery_col_312
        #     battery_col_312 = "Ultium 312"  # Clean up any whitespace
        #     # Mask for 7/1 to 7/5 (2025) with explicit UTC localization
        #     start_dt = pd.to_datetime("2025-07-01").tz_localize("UTC")
        #     end_dt = pd.to_datetime("2025-07-04").tz_localize("UTC")
        #     mask = (analysis_df["timestamp"] >= start_dt) & (analysis_df["timestamp"] < end_dt)
        #     df_masked = analysis_df.loc[mask]
        #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        #     ax1.plot(
        #         df_masked["timestamp"],
        #         df_masked[bat_name],
        #         label=f"Feedrate of {battery_col_312} per hour",
        #         color="tab:blue",
        #     )
        #     ax1.set_ylabel(f"Feedrate {battery_col_312}", fontsize=14)
        #     ax1.legend(loc="upper right")
        #     ax1.grid(True)

        #     ax2.plot(df_masked["timestamp"], df_masked["motor_amps"], label="Motor Amps", color="tab:orange")
        #     ax2.set_ylabel("Motor Amps / Rev", fontsize=14)
        #     ax2.set_xlabel("Timestamp", fontsize=14)
        #     ax2.legend(loc="upper right")
        #     ax2.grid(True)

        #     # Plot RPM (left y-axis, red) and Kiln Weight (right y-axis, green) on the same subplot
        #     ax3a = ax3
        #     ax3b = ax3.twinx()
        #     l1 = ax3a.plot(df_masked["timestamp"], df_masked["rpm"], label="RPM", color="tab:red")
        #     l2 = ax3b.plot(df_masked["timestamp"], df_masked["kiln_weight"], label="Kiln Weight", color="tab:green")
        #     ax3a.set_ylabel("RPM", fontsize=14, color="tab:red")
        #     ax3b.set_ylabel("Kiln Weight", fontsize=14, color="tab:green")
        #     ax3a.tick_params(axis="y", labelcolor="tab:red")
        #     ax3b.tick_params(axis="y", labelcolor="tab:green")
        #     # Combine legends
        #     lines = l1 + l2
        #     labels = [line.get_label() for line in lines]
        #     ax3a.legend(lines, labels, loc="upper right")
        #     ax3a.grid(True)

        #     plt.suptitle(f"Time Series (7/1-7/5): {battery_col_312} Feedrate and Motor Amps", fontsize=18)
        #     plt.tight_layout(rect=(0, 0.03, 1, 0.97))
        #     plt.show()
        # else:
        #     print("No battery column containing '312' found in analysis_df.")

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
        if STANDARDIZE:
            print("\nRunning standardised regression...")
            model_std, beta_tbl = run_standardised_regression(
                analysis_df=analysis_df,
                base_predictors=base_predictors,
                category_predictors=investigation_cols,
                print_results=True,
            )
        else:
            print("\nRunning regular regression...")
            model_std, beta_tbl = run_multiple_linear_regression(
                analysis_df=analysis_df,
                base_predictors=base_predictors,
                category_predictors=investigation_cols,
                print_results=True,
            )

        # Add standardized flag to filename if applicable
        standardized_suffix = "_standardized" if STANDARDIZE else ""

        # Write standardised regression coefficients to CSV
        regression_filename = f"{results_folder}/scout_{analysis_type}{standardized_suffix}_regression_impacts.csv"
        beta_tbl.to_csv(regression_filename, index=True)
        print(f"Standardised regression results written to {regression_filename}")


if __name__ == "__main__":
    run()
