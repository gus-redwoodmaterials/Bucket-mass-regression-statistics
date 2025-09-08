import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def run():
    material_trans_data = pd.read_csv(
        "/Users/gus.robinson/Desktop/Local Python Coding/Kiln Modelling/Data/material_trans_data.csv"
    )
    ms4_mass = material_trans_data[
        material_trans_data["location_description"] == "MS4 Stored Metal Concentrate Tipper"
    ]["quantity"].sum()  # kgs
    print(f"MS4 Mass: {ms4_mass} kgs")
    infeed_mass = material_trans_data[
        material_trans_data["location_description"] != "MS4 Stored Metal Concentrate Tipper"
    ]["quantity"].sum()  # kgs
    print(f"Infeed Mass: {infeed_mass} kgs")
    print(f"Ratio (MS4 / Infeed): {ms4_mass / infeed_mass:.2%}")
    return 0


if __name__ == "__main__":
    run()
