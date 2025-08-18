import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os


def run():
    results_folder = "Current Spike/results"
    all_files = [f for f in os.listdir(results_folder) if f.endswith(".csv")]
    dfs = {}
    for fname in all_files:
        fpath = os.path.join(results_folder, fname)
        df = pd.read_csv(fpath)
        if "material" in fname or "two_stage" not in fname:
            continue
        if "standardized" in fname:
            # Rename 'beta_std' to 'impact' if present
            df = df.rename(columns={"beta_std": "impact"})
        else:
            continue
        dfs[fname] = df

    print(dfs.keys())
    impacts = {}
    ignore = []
    for fname, df in dfs.items():
        if "var" not in df.columns or "impact" not in df.columns:
            continue
        for idx, row in df.iterrows():
            var = str(row["var"])
            if var.startswith("r100"):
                prefix = var.split("___", 1)[1] if "___" in var else var
                if prefix == "ultium_pouch":
                    prefix = "ultium_312"
            else:
                prefix = var
            if prefix in ignore:
                continue
            impact_val = float(row["impact"])
            p_value = row.get("p_value", None)
            if impact_val < 0:
                ignore.append(prefix)
                if prefix in impacts:
                    del impacts[prefix]
                continue
            if p_value is not None and p_value > 0.05:
                ignore.append(prefix)
                if prefix in impacts:
                    del impacts[prefix]
                continue
            if prefix not in impacts:
                impacts[prefix] = []
            impacts[prefix].append((impact_val, p_value))

    # --- Bar chart of standardized regression impacts for all variables ---

    # Collect all impacts and variable names from standardized files
    battery_stats = {}
    for battery in list(impacts.keys()):
        vals = impacts.get(battery)
        if vals is None or len(vals) == 0:
            del impacts[battery]
            continue
        impact_vals = [v[0] for v in vals]
        p_vals = [v[1] for v in vals if v[1] is not None]
        mean_impact = np.mean(impact_vals)
        mean_p_value = np.mean(p_vals) if p_vals else None
        if mean_impact > 10:
            del impacts[battery]
            continue
        battery_stats[battery] = {"mean_impact": mean_impact, "mean_p_value": mean_p_value}

    filtered = [(b, s["mean_impact"], s["mean_p_value"]) for b, s in battery_stats.items()]
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_vars, top_impacts, top_pvals = zip(*filtered[:7]) if filtered else ([], [], [])

    # Write top 7 batteries to CSV
    top_battery_df = pd.DataFrame({"battery": top_vars, "mean_impact": top_impacts, "mean_p_value": top_pvals})
    top_battery_csv = os.path.join(results_folder, "top7_batteries_by_impact.csv")
    top_battery_df.to_csv(top_battery_csv, index=False)
    print(f"Top 7 batteries written to {top_battery_csv}")

    # "#2ca02c"
    plt.figure(figsize=(8, 6))
    bar_colors = ["#d62728" if str(v).startswith("r1") else "#d62728" for v in top_vars]
    bars = plt.bar(top_vars, top_impacts, color=bar_colors)
    plt.ylabel("Amps per $\sigma$")
    plt.xlabel("Variable")
    plt.title("Top 7 Correlated Variables")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Add legend for color meaning
    # from matplotlib.patches import Patch

    # legend_handles = [
    #     Patch(color="#2ca02c", label="Physical"),
    #     Patch(color="#d62728", label="Batteries"),
    # ]
    # plt.legend(handles=legend_handles, loc="best")
    plt.show()


if __name__ == "__main__":
    run()
