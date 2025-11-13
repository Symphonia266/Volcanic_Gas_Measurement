import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from . import package_path
from ..pasquill_stable_classfication import (
    classification_table,
    classification_label,
    classify_atomosphere_stability,
    inverse_stab_class_to_wether
)

lateral = {
    "A": pd.read_csv(
        os.path.join(package_path, "data", "sampling_lateral", "A_extremely_unstable.csv"),
        names=["dist", "spread"],
    ),
    "B": pd.read_csv(
        os.path.join(package_path, "data", "sampling_lateral", "B_moderately_unstable.csv"),
        names=["dist", "spread"],
    ),
    "C": pd.read_csv(
        os.path.join(package_path, "data", "sampling_lateral", "C_slightly_unstable.csv"),
        names=["dist", "spread"],
    ),
    "D": pd.read_csv(
        os.path.join(package_path, "data", "sampling_lateral", "D_neutral.csv"),
        names=["dist", "spread"],
    ),
    "E": pd.read_csv(
        os.path.join(package_path, "data", "sampling_lateral", "E_slightly_stable.csv"),
        names=["dist", "spread"],
    ),
    "F": pd.read_csv(
        os.path.join(package_path, "data", "sampling_lateral", "F_moderately_stable.csv"),
        names=["dist", "spread"],
    ),
}
vertical = {
    "A": pd.read_csv(
        os.path.join(package_path, "data", "sampling_vertical", "A_extremely_unstable.csv"),
        names=["dist", "spread"],
    ),
    "B": pd.read_csv(
        os.path.join(package_path, "data", "sampling_vertical", "B_moderately_unstable.csv"),
        names=["dist", "spread"],
    ),
    "C": pd.read_csv(
        os.path.join(package_path, "data", "sampling_vertical", "C_slightly_unstable.csv"),
        names=["dist", "spread"],
    ),
    "D": pd.read_csv(
        os.path.join(package_path, "data", "sampling_vertical", "D_neutral.csv"),
        names=["dist", "spread"],
    ),
    "E": pd.read_csv(
        os.path.join(package_path, "data", "sampling_vertical", "E_slightly_stable.csv"),
        names=["dist", "spread"],
    ),
    "F": pd.read_csv(
        os.path.join(package_path, "data", "sampling_vertical", "F_moderately_stable.csv"),
        names=["dist", "spread"],
    ),
}


def pg_chart_interp(xy):
    obs_x = np.log10(xy["dist"])
    obs_y = np.log10(xy["spread"])
    log_y = interp1d(obs_x, obs_y, bounds_error=False, fill_value=np.nan)
    y = lambda x: np.power(10, log_y(np.log10(x)))
    return y


if __name__ == "__main__":
    i = np.linspace(0, 5, 10000)
    x = np.power(10, i)

    sigma_y = {}
    sigma_z = {}
    for lb in classification_label:
        f_y = pg_chart_interp(lateral[lb])
        f_z = pg_chart_interp(vertical[lb])
        sigma_y[lb] = f_y(x)
        sigma_z[lb] = f_z(x)
    print(sigma_y["A"])
    print(sigma_z["A"])

    df1 = pd.DataFrame(sigma_y)
    df2 = pd.DataFrame(sigma_z)
    df1.insert(0, "dist", x)
    df2.insert(0, "dist", x)
    df1.to_csv(package_path + "normalized_lateral_spreadwidth.csv")
    df2.to_csv(package_path + "normalized_vertical_spreadwidthe.csv")

    fig, ax = plt.subplots(1, 2, layout="tight")
    ax[0].grid(which="major", axis="both", ls="-", c="darkgrey")
    ax[1].grid(which="major", axis="both", ls="-", c="darkgrey")
    ax[0].grid(which="minor", axis="both", ls="--", c="lightgrey")
    ax[1].grid(which="minor", axis="both", ls="--", c="lightgrey")
    for lb in classification_label:
        ax[0].plot(x, sigma_y[lb], label=lb)
        ax[1].plot(x, sigma_z[lb], label=lb)
        i += 1
    # ax[0].plot(data_B["dist"], data_B["spread"], label="B")
    # ax[0].plot(data_C["dist"], data_C["spread"], label="C")
    # ax[0].plot(data_D["dist"], data_D["spread"], label="D")
    # ax[0].plot(data_E["dist"], data_E["spread"], label="E")
    # ax[0].plot(data_F["dist"], data_F["spread"], label="F")
    ax[0].set_title("Plot the points to create the data.")
    ax[0].set_xlabel("downwind distance [m]")
    ax[0].set_ylabel("lateral spread [m]")
    # ax[0].set_xscale("log")
    # ax[0].set_yscale("log")
    ax[0].set_xlim(1e2, 1e5)
    ax[0].set_ylim(4, 1e4)

    ax[1].set_title("linear interp by the created data.")
    ax[1].set_xlabel("downwind distance [m]")
    ax[1].set_ylabel("lateral spread [m]")
    # ax[1].set_xscale("log")
    # ax[1].set_yscale("log")
    ax[1].set_xlim(1e2, 1e5)
    ax[1].set_ylim(1, 3e3)

    plt.show(block=False)
    input()
