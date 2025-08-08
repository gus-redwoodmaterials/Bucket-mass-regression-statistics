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
        if "material" in fname or "cumulative" not in fname:
            continue
        if "standardized" in fname:
            # Rename 'beta_std' to 'impact' if present
            df = df.rename(columns={"beta_std": "impact"})
        else:
            continue
        dfs[fname] = df

    print(dfs.keys())
    # Aggregate r100 variables across all standardized files
    impacts = {}
    ignore = []
    for fname, df in dfs.items():
        if "var" not in df.columns or "impact" not in df.columns:
            continue
        for idx, row in df.iterrows():
            var = str(row["var"])
            if var.startswith("r100"):
                prefix = var.split("_")[0] if "_" in var else var
            else:
                prefix = var
            if prefix in ignore:
                continue
            if prefix not in impacts:
                impacts[prefix] = []
                try:
                    impact_val = float(row["impact"])
                    if impact_val < 0:
                        ignore.append(prefix)
                        if prefix in impacts:
                            del impacts[prefix]
                        continue
                    p_value = row.get("p_value", None)
                    if p_value is not None and p_value > 0.05:
                        continue
                    impacts[prefix].append(impact_val)
                except Exception:
                    pass

    # --- Bar chart of standardized regression impacts for all variables ---

    # Collect all impacts and variable names from standardized files
    for battery in list(impacts.keys()):
        if impacts.get(battery) is None:
            continue
        if len(impacts[battery]) > 0:
            impacts[battery] = np.median(impacts[battery])
        else:
            del impacts[battery]
            continue
        if impacts[battery] > 10:
            del impacts[battery]

    filtered = list(impacts.items())
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_vars, top_impacts = zip(*filtered[:7]) if filtered else ([], [])

    plt.figure(figsize=(8, 6))
    bar_colors = ["#d62728" if str(v).startswith("r1") or str(v).startswith("u") else "#2ca02c" for v in top_vars]
    bars = plt.bar(top_vars, top_impacts, color=bar_colors)
    plt.ylabel("Amps per $\sigma$")
    plt.xlabel("Variable")
    plt.title("Top 7 Correlated Variables")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Add legend for color meaning
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="#2ca02c", label="Physical"),
        Patch(color="#d62728", label="Batteries"),
    ]
    plt.legend(handles=legend_handles, loc="best")
    plt.show()


if __name__ == "__main__":
    run()
