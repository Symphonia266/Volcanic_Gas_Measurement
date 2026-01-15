import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from gas_simulation import utils
from gas_simulation.model import Environment
from gas_simulation.lidar_model.utils import Coord
from gas_simulation.lidar_model import lidar
from gas_simulation.lidar_model import dial

def lidar_equation_result_viewer(
    wl: np.ndarray,
    lidar_coord: Coord,
    debug: dial.DialDebugData,
    dist_idx: int = -1,
    wl_idx: int = 0,
):
    fig, axes = plt.subplots(2, 2, layout="constrained", sharex="col")
    axes[0, 0].plot(
        lidar_coord.distance, debug.tau_on[:, wl_idx], c="red", ls="-", label="on"
    )
    axes[0, 0].plot(
        lidar_coord.distance, debug.tau_off[:, wl_idx], c="blue", ls="-", label="off"
    )
    # axes[0].plot(coord.distance[1:], tau_N2[:, idx_300nm],  c ="orange",ls="--", label="N2")
    # axes[0].plot(coord.distance[1:], tau_O2[:, idx_300nm],  c ="green", ls="--", label="O2")

    axes[1, 0].plot(
        lidar_coord.distance,
        debug.p_on[:, wl_idx],
        marker="o",
        c="red",
        ls="-",
        label="on",
    )
    axes[1, 0].plot(
        lidar_coord.distance,
        debug.p_off[:, wl_idx],
        marker="o",
        c="blue",
        ls="-",
        label="off",
    )

    axes[0, 1].plot(wl, debug.tau_on[dist_idx, :], c="red", ls="-", label="on")
    axes[0, 1].plot(wl, debug.tau_off[dist_idx, :], c="blue", ls="-", label="off")

    axes[1, 1].plot(wl, debug.p_on[dist_idx, :], c="red", ls="-", label="on")
    axes[1, 1].plot(wl, debug.p_off[dist_idx, :], c="blue", ls="-", label="off")

    axes[1, 0].set_xlabel("lidar distance [m]")
    axes[1, 1].set_xlabel("laser wavelength [nm]")
    axes[0, 0].set_ylabel(r"transmittance $\tau$")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 1].set_ylim(0, 1.0)

    axes[1, 0].set_ylabel(r"receive power $P_{phot}$")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")

    for ax in axes.flatten():
        ax.grid(which="major", ls="-", c="darkgrey")
        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.legend()

    return fig, axes


def dial_equation_result_viewer(
    env: Environment,
    wl: np.ndarray,
    results: dial.DialResult,
    cf: np.ndarray,
    dist_idx: int = -1,
    wl_idx: int = 0,
):
    fig = plt.figure(layout="constrained")
    gs = GridSpec(
        2,
        2,
        width_ratios=[1, 1],  # 左にプロット、右にカラーバー
        height_ratios=[1, 1],  # 上下等分
        # wspace=0.0,           # 左右の余白最小
        # hspace=0.05             # 上下の余白ゼロ
        figure=fig,
    )
    ax_dist = fig.add_subplot(gs[:, 0])
    ax_err1_wl = fig.add_subplot(gs[0, 1])
    ax_err2_wl = fig.add_subplot(gs[1, 1], sharex=ax_err1_wl)
    axes = [ax_dist, ax_err1_wl, ax_err2_wl]

    r = np.linspace(results.coord.distance.min(), results.coord.distance.max(), 1000)
    x = np.linspace(results.coord.x.min(), results.coord.x.max(), 1000)
    z = np.linspace(results.coord.z.min(), results.coord.z.max(), 1000)
    ax_dist.plot(
        r,
        utils.number_density_to_ppm(env.number_density_at(x, 0, z)["SO2"], z),
        c="black",
        alpha=0.5,
        label="Enveronment True",
    )
    ax_dist.scatter(
        results.coord.distance,
        utils.number_density_to_ppm(results.n_true["SO2"], results.coord.z),
        marker="*",
        s=100,
        c="black",
        label="DIAL True",
    )
    ax_dist.errorbar(
        x=results.coord.distance,
        y=utils.number_density_to_ppm(results.res[:, wl_idx], results.coord.z),
        yerr=utils.number_density_to_ppm(results.stat_err[:, wl_idx], results.coord.z),
        capsize=8,
        fmt="o",
        markersize=6,
        ecolor="red",
        color="red",
        label="DIAL Result",
    )
    ax_dist.errorbar(
        x=results.coord.distance,
        y=utils.number_density_to_ppm(
            (results.res - cf)[:, wl_idx], results.coord.z
        ),
        yerr=utils.number_density_to_ppm(results.stat_err[:, wl_idx], results.coord.z),
        capsize=8,
        fmt="o",
        markersize=6,
        ecolor="green",
        color="green",
        label="DIAL Result (mol. and aer. Correction)",
    )
    ax_dist.set(
        xlabel="(Line of Sight) distance [m]",
        ylabel=r"concentration $n_{SO2}$[ppm]",
        xlim=(results.coord.x.min(), results.coord.x.max()),
        # ylim=(0, None)
    )
    ax_dist.legend()

    ax_err1_wl.scatter(
        wl,
        np.abs(
            (results.res[dist_idx, :] - results.n_true["SO2"][dist_idx])
            / results.n_true["SO2"][dist_idx]
        ),
        c="red",
        label="DIAL Result",
    )
    ax_err1_wl.scatter(
        wl,
        np.abs(
            100
            * (
                (results.res - cf)[dist_idx, :]
                - results.n_true["SO2"][dist_idx]
            )
            / results.n_true["SO2"][dist_idx]
        ),
        c="green",
        label="DIAL Result (mol. and aer. Correction)",
    )
    ax_err1_wl.set(
        xlabel=r"laser wavelength [nm]",
        ylabel=r"|contamination error| [%]",
        yscale="log",
    )
    ax_err1_wl.legend()

    ax_err2_wl.scatter(
        wl,
        np.abs(100 * results.stat_err[dist_idx, :] / results.n_true["SO2"][dist_idx]),
        c="black",
    )
    ax_err2_wl.set(
        xlabel=r"laser wavelength [nm]",
        ylabel=r"|statistical error| [%]",
        yscale="log",
    )

    for ax in axes:
        ax.grid(which="major", ls="-", c="darkgrey")
        ax.grid(which="minor", ls="--", c="lightgrey")

    return fig, axes
