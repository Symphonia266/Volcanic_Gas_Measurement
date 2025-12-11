import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec 

from gas_simulation import utils
from gas_simulation.model import PlumeEnvironment 
from gas_simulation.lidar_model.lidar import Coord 

def lidar_equation_result_viewer(
        coord:Coord, 
        tau_on_dist, 
        tau_off_dist, 
        p_on_dist, 
        p_off_dist, 

        wl, 
        tau_on_wl, 
        tau_off_wl,
        p_on_wl,
        p_off_wl,
    ): 
    fig, axes = plt.subplots(2, 2, layout="constrained", sharex="row")
    axes[0, 0].plot(coord.distance, tau_on_dist, c="red", ls="-", label="on")
    axes[0, 0].plot(coord.distance, tau_off_dist, c="blue", ls="-", label="off")
    # axes[0].plot(coord.distance[1:], tau_N2[:, idx_300nm],  c ="orange",ls="--", label="N2")
    # axes[0].plot(coord.distance[1:], tau_O2[:, idx_300nm],  c ="green", ls="--", label="O2")

    axes[0, 1].plot(coord.distance, p_on_dist,  marker="o", c="red", ls="-", label="on")
    axes[0, 1].plot(coord.distance, p_off_dist, marker="o", c="blue", ls="-", label="off")

    axes[1, 0].plot(wl, tau_on_wl, c="red", ls="-", label="on")
    axes[1, 0].plot(wl, tau_off_wl, c="blue", ls="-", label="off")
    # axes[0].plot(wl_laser, tau_N2[9, :],  c ="orange",ls="--", label="N2")
    # axes[0].plot(wl_laser, tau_O2[9, :],  c ="green", ls="--", label="O2")

    axes[1, 1].plot(wl, p_on_wl, c="red", ls="-", label="on")
    axes[1, 1].plot(wl, p_off_wl, c="blue", ls="-", label="off")
    # axes[1].plot(wl_laser, p_N2[9, :],  c ="orange",ls="--", label="N2")
    # axes[1].plot(wl_laser, p_O2[9, :],  c ="green", ls="--", label="O2")

    for ax in axes[0, :]:
        ax.set_xlabel("lidar distance [m]")
    for ax in axes[1, :]:
        ax.set_xlabel("laser wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"transmittance $\tau$")
        ax.set_ylim(0, 1.0)
    for ax in axes[:, 1]:
        ax.set_ylabel(r"receive power $P_{phot}$")
        ax.set_yscale("log")

    for ax in axes.flatten():
        ax.grid(which="major", ls="-", c="darkgrey")
        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.legend()

    return fig, axes

def dial_equation_result_viewer(
    env:PlumeEnvironment, 

    coord:Coord,
    n_true_dist,
    res1_dist, 
    res2_dist,
    stat_err_dist, 

    wl,
    n_true_wl:float,
    res1_wl, 
    res2_wl, 
    stat_err_wl,
):
    fig = plt.figure(layout="constrained")
    gs = GridSpec(
        2, 2,
        width_ratios=[1, 1],   # 左にプロット、右にカラーバー
        height_ratios=[1, 1],  # 上下等分
        # wspace=0.0,           # 左右の余白最小
        # hspace=0.05             # 上下の余白ゼロ
        figure=fig
    )
    ax_dist = fig.add_subplot(gs[:, 0])
    ax_err1_wl = fig.add_subplot(gs[0, 1])
    ax_err2_wl = fig.add_subplot(gs[1, 1], sharex=ax_err1_wl)
    axes = [ax_dist, ax_err1_wl, ax_err2_wl]

    r = np.linspace(coord.distance.min(), coord.distance.max(), 1000)
    x = np.linspace(coord.x.min(), coord.x.max(), 1000)
    z = np.linspace(coord.z.min(), coord.z.max(), 1000)
    ax_dist.scatter(
        coord.distance,
        utils.number_density_to_ppm(n_true_dist, coord.z),
        marker="*",
        s=100,
        c="black",
        label="DIAL True"
    )
    ax_dist.errorbar(
        x=coord.distance,
        y=utils.number_density_to_ppm(res1_dist, coord.z),
        yerr=utils.number_density_to_ppm(stat_err_dist, coord.z),
        capsize=8, fmt='o', markersize=6, ecolor='red', color='red',
        label="DIAL Result",
    )
    ax_dist.errorbar(
        x=coord.distance,
        y=utils.number_density_to_ppm(res2_dist, coord.z),
        yerr=utils.number_density_to_ppm(stat_err_dist, coord.z),
        capsize=8, fmt='o', markersize=6, ecolor='green', color='green',
        label="DIAL Result (mol. and aer. Correction)",
    )
    ax_dist.plot(
        r,
        utils.number_density_to_ppm(env.number_density_at(x, 0, z)["SO2"], z),
        c="black",
        alpha = 0.5,
        label="Enveronment True",
    )
    ax_dist.set_xlabel("lidar distance [m]")
    ax_dist.set_ylabel(r"concentration $n_{SO2}$[ppm]")
    ax_dist.legend()

    ax_err1_wl.scatter(wl, np.abs(res1_wl/n_true_wl), c="red", label="DIAL Result")
    ax_err1_wl.scatter(wl, np.abs(res2_wl/n_true_wl), c="green", label="DIAL Result (mol. and aer. Correction)")
    ax_err1_wl.set_xlabel(r"laser wavelength [nm]")
    ax_err1_wl.set_ylabel(r"|contamination error| [%]")
    ax_err1_wl.set_yscale("log")
    ax_err1_wl.legend()

    ax_err2_wl.scatter(wl, np.abs(stat_err_wl/n_true_wl), c="black")
    ax_err2_wl.set_xlabel(r"laser wavelength [nm]")
    ax_err2_wl.set_ylabel(r"|statistical error| [%]")
    ax_err2_wl.set_yscale("log")

    for ax in axes:
        ax.grid(which="major", ls="-", c="darkgrey")
        ax.grid(which="minor", ls="--", c="lightgrey")

    return fig, axes
