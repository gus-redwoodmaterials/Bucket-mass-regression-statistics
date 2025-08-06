from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz

from dateutil.parser import parse
from rw_data_science_utils.athena_download import athena_download


TAGS = {
    "bucket_mass": "RC1/4420-Calciner/AcmeStatus/bucket_mass".lower(),
}

# style copied from gather_test_data
TABLE_PATH = "cleansed.rc1_historian"
pacific_tz = pytz.timezone("US/Pacific")

START = pacific_tz.localize(parse("2025-07-20T20:00:00"))
END = pacific_tz.localize(parse("2025-07-21T08:00:00"))

DATA_TIMESTEP_SECONDS = 3


def run():
    df = athena_download.get_pivoted_athena_data(
        TAGS,
        START,
        END,
        TABLE_PATH,
        DATA_TIMESTEP_SECONDS,
    )

    # breakpoint()

    t = df["timestamp"]

    # Need to retroactively apply fixed scaling
    for i in np.arange(6) + 1:
        df[f"t_to_refr_{i}"] *= (1176.67 + 17.7) / (df["t_to_refr_maxpoint"] + 17.7)

    _, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(t, df["t_to"], color="b", label="TO gas internal (TT-0003A)")
    ax[0].plot(t, df["t_to_refr_1"], color="grey", label="Refractory TCs (TT-0026-31)")
    ax[0].plot(t, df["t_to_refr_2"], color="grey")
    ax[0].plot(t, df["t_to_refr_3"], color="grey")
    ax[0].plot(t, df["t_to_refr_4"], color="grey")
    ax[0].plot(t, df["t_to_refr_5"], color="grey")
    ax[0].plot(t, df["t_to_refr_6"], color="grey")
    ax[0].plot(t, df["t_baghouse_a"], color="m", label="Baghouse TCs (TT-0010-13)")
    ax[0].plot(t, df["t_baghouse_b"], color="m")
    ax[0].plot(t, df["t_baghouse_c"], color="m")
    ax[0].plot(t, df["t_baghouse_d"], color="m")

    ax[0].set_ylabel("Deg C")
    ax[0].legend()
    ax[0].grid("on")

    ax[1].plot(t, df["sp1_cc"], color="k", label="CC SP1")
    ax[1].plot(t, df["sp2_cc"], color="r", label="CC SP2")

    ax[1].set_ylabel("%")
    ax[1].legend()
    ax[1].grid("on")

    ax[2].plot(t, df["hz_main_fan"], color="k", label="Main Fan Hz (FAN_004)")

    ax[2].set_ylabel("Hz")
    ax[2].legend()
    ax[2].grid("on")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
