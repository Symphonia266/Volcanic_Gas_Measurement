# ================================
# Standard library imports
# ================================
import sys
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass

# ================================
# Third-party imports
# ================================
from matplotlib.pylab import xscale
import numpy as np
import pandas as pd
from matplotlib import cm, ticker
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter

from numpy.lib.stride_tricks import sliding_window_view as np_SWV

# ================================
# Project path setup
# ================================
# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# ================================
# Project-specific imports
# ================================
from gas_simulation import utils
from gas_simulation.consts import main_gases_props
from gas_simulation.atom import betas_N2, betas_O2
from gas_simulation.diffusion_model import pasquill_stable_classfication as PSC

from gas_simulation.model import (
    Field,
    Source,
    gen_fauntainsource,
    Gas,
    PlumeEnvironment,
)
from gas_simulation.lidar_model.utils import Coord
from gas_simulation.lidar_model import lidar
from gas_simulation.lidar_model import dial
from gas_simulation import result_viewer as viewer

# ================================
# Matplotlib global style
# ================================
plt.style.use("my_sty.mplstyle")

# ================================
# Constants
# ================================
T_SEC = 60 * 30


# ================================
# Utility functions
# ================================
def fig_with_fixed_ax(
    ax_size: tuple[float, float], left=0.15, right=0.95, bottom=0.1, top=0.98
):
    fig_w = ax_size[0] / (right - left)
    fig_h = ax_size[1] / (top - bottom)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    return fig, ax


def plt_log_mode(ax):
    ax.set_yscale("log")
    # plt.grid(which='major',color='black',linestyle='-')
    # plt.grid(which='minor',color='lightgrey',linestyle='--', axis="y")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.grid(which="minor", ls="--", c="lightgrey", axis="y")
    return ax


lbs = lambda s1, s2: f"{s1}, {s2}"

# ================================
# Cross sections
# ================================
xs_SO2 = utils.load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)
xs_H2S = utils.load_cross_section(
    "H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)
xs_O3 = utils.load_cross_section(
    "O3_Bogumil(2003)_293K_230-1070nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)

# ================================
# Lidar / DIAL setup
# ================================
lc = lidar.LidarCalc(M=100 * T_SEC, dR=1.5)
dc = dial.DialCalc()
lc.show_params()
dc.show_params()

lidar_coord = Coord(
    distance=np.arange(5, 200, 5),
    theta_deg=0.0,
    x0=0,
    z0=1000,
)

# ================================
# Environment definition
# ================================
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -10], N_pt=30)
# field = Field(2, weather="clear", wind_direction_deg=90)
field = Field(10, weather="cloudy", wind_direction_deg=90)
src = np.array(
    [
        [1, 50, -10, lidar_coord.z0 + 1],
        [1, 100, -30, lidar_coord.z0 + 1],
        [1, 150, -50, lidar_coord.z0 + 1],
        # [1, 600, -20, lidar_coord.z0 + 2],
        # [1, 625, -40, lidar_coord.z0 + 2],
        # [1, 650, -20, lidar_coord.z0 + 2],
    ]
).T
source = Source(src[0], src[1], src[2], src[3])
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[300, -50], N_pt=10)
# He = np.full_like(q, lidar_coord.z0 + 2)
# source = Source(q, x_src, y_src, He)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[500, -100], N_pt=10)
# He = np.full_like(q, lidar_coord.z0 + 2)
# source.add(q, x_src, y_src, He)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[700, -50], N_pt=10)
# He = np.full_like(q, lidar_coord.z0 + 2)
# source.add(q, x_src, y_src, He)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[725, -100], N_pt=10)
# He = np.full_like(q, lidar_coord.z0 + 2)
# source.add(q, x_src, y_src, He)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[750, -50], N_pt=10)
# He = np.full_like(q, lidar_coord.z0 + 2)
# source.add(q, x_src, y_src, He)
env = PlumeEnvironment(
    field=field,
    source=source,
    time=T_SEC,
    gas={
        # 煙源直下で30ppmになるよう調整したQ
        # "SO2": Gas(Q=80e5, offset=0, cross_section=xs_SO2),
        # "H2S": Gas(Q=40e5, offset=0, cross_section=xs_H2S),
        # 今回の煙源位置、風プロファイル設定で30ppm程度になるよう調整したQ
        "SO2": Gas(Q=30e6, offset=0, cross_section=xs_SO2),
        "H2S": Gas(Q=7.5e6, offset=0, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
)
fig_env1, _, fig_env2, _ = env.show_gases(lidar_coord)

# ================================
# Wavelength definitions
# ================================
wl_laser = np.arange(240, 370, 0.02)
wl_laser = np.arange(240, 370, 0.02)
wl = {
    "laser": wl_laser,
    "N2_st": utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], False),
    "O2_st": utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], False),
    "N2_as": utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], True),
    "O2_as": utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], True),
}

# ================================
# Backscatter and transmission
# ================================
beta_N2 = betas_N2(lidar_coord.z)
beta_O2 = betas_O2(lidar_coord.z)
tau = {k:  env.transmittance(lidar_coord, v) for k, v in wl.items()}
beta_tau = {
    "N2_st": beta_N2[:, np.newaxis] * tau["laser"] * tau["N2_st"],
    "O2_st": beta_O2[:, np.newaxis] * tau["laser"] * tau["O2_st"],
    "N2_as": beta_N2[:, np.newaxis] * tau["laser"] * tau["N2_as"] * 0.1,
    "O2_as": beta_O2[:, np.newaxis] * tau["laser"] * tau["O2_as"] * 0.1,
}

# ================================
# Power calculation
# ================================
p = {
    k: lc.power(
        dist=lidar_coord.distance[:, np.newaxis],
        wl=wl_laser[np.newaxis, :],
        beta_tau=v,
    )
    for k, v, in beta_tau.items()
}

# ================================
# DIAL inputs
# ================================
pair_data = {
    (s1, s2): dial.DialInput(
        env=env,
        obj_s1=dial.RamanShiftObject(
            wl=wl[s1],
            xs=env.gas_inventory["SO2"].cross_section(wl[s1]),
            p=p[s1],
        ),
        obj_s2=dial.RamanShiftObject(
            wl=wl[s2],
            xs=env.gas_inventory["SO2"].cross_section(wl[s2]),
            p=p[s2],
        ),
        lidar_coord=lidar_coord,
    )
    for s1, s2 in combinations(p.keys(), 2)
}

# ================================
# Results containers
# ================================
results: dict[tuple[str, str], dial.DialResult] = {}
cfs: dict[tuple[str, str], np.ndarray] = {}
debug: dict[tuple[str, str], dial.DialDebugData] = {}
analysis: pd.DataFrame = pd.DataFrame(
    [],
    columns=[
        "idx",
        "wl_ls",
        "wl_on",
        "wl_off",
        "stat_err_ppm",
        "contam_err_ppm",
        "d_xs",
    ],
)

# ================================
# Main DIAL loop
# ================================
for (s1, s2), dial_input in pair_data.items():
    print(f"calculation ({s1}, {s2}) combination ramanDIAL...")

    # === DIAL calculation ===
    results[(s1, s2)], cf_input, debug[(s1, s2)] = dc.estimate(dial_input)
    cfs[(s1, s2)] = dial.calc_correction_factor(cf_input, mol=True, aer=True)
    idx_wl_trgt = np.nanargmin(np.abs(results[(s1, s2)].stat_err[-1, :]))

    analysis.loc[f"{s1}-{s2}"] = pd.Series(
        {
            "idx": idx_wl_trgt,
            "wl_ls": wl_laser[idx_wl_trgt],
            "wl_on": results[(s1, s2)].wl_on[idx_wl_trgt],
            "wl_off": results[(s1, s2)].wl_off[idx_wl_trgt],
            "stat_err_ppm": utils.number_density_to_ppm(
                results[(s1, s2)].stat_err[-1, idx_wl_trgt],
                results[(s1, s2)].coord.z[-1],
            ),
            "contam_err_ppm": utils.number_density_to_ppm(
                results[(s1, s2)].res[-1, idx_wl_trgt] - cfs[(s1, s2)][-1, idx_wl_trgt],
                results[(s1, s2)].coord.z[-1],
            ),
            "d_xs": debug[(s1, s2)].d_xs[idx_wl_trgt],
        }
    )
analysis["rank"] = np.argsort(np.argsort(analysis[["stat_err_ppm"]].values.flatten()))
print(analysis.sort_values(by="rank"))

# ================================
# Result Printing (Raw text)
# ================================
# for (s1, s2), res_obj in results.items():
#     # print(f"最遠方濃度{utils.number_density_to_ppm(res_obj.n_true["SO2"][-1], res_obj.coord.z[-1]):.3f} [ppm]")
#     lb = f"{s1}-{s2}"
#     i = analysis.at[lb, "rank"]
#     idx_trgt = int(analysis.at[lb, "idx"])
#     # idx_trgt = np.searchsorted(wl_laser, 320.0)
#     cond = analysis.at[lb, "wl_on"] < analysis.at[lb, "wl_off"]
#     txt_wl_ls = f"{analysis.at[lb, "wl_ls"]:.2f}"
#     txt_on = f"on :{analysis.at[lb, "wl_on"]:.1f} nm"
#     txt_off = f"off:{analysis.at[lb, "wl_off"]:.1f} nm"
#     a:str = txt_on if cond else txt_off
#     b:str = txt_off if cond else txt_on
#     txt_staterr = f"{analysis.at[lb, 'stat_err_ppm']:.4f} [ppm]"

# ================================
# Plotting
# ================================

fig_pasq_l, ax_pasq_l = plt.subplots(figsize=(3,3))
fig_pasq_v, ax_pasq_v = plt.subplots(figsize=(3,3))
fig = plt.figure()
gs = GridSpec(
    2,
    2,
    width_ratios=[2, 1],  # 左に距離特性、右に波長特性
    height_ratios=[1, 1],  # 上下等分
    # wspace=0.0,           # 左右の余白最小
    # hspace=0.05             # 上下の余白ゼロ
    figure=fig,
)
ax_dist = fig.add_subplot(gs[:, 0])
ax_err1_wl = fig.add_subplot(gs[0, 1])
ax_err2_wl = fig.add_subplot(gs[1, 1], sharex=ax_err1_wl)
axes = [ax_dist, ax_err1_wl, ax_err2_wl]
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

# ax3_inset1をax3内に用意する
ax3_ins1 = ax3.inset_axes([0.45, 0.45, 0.52, 0.52])
# ax3_ins2 = ax3.inset_axes([0.6, 0.3, 0.3, 0.6])
ax3.grid(False)
ax3_ins1.grid(False)

plt_log_mode(ax_err1_wl)
plt_log_mode(ax_err2_wl)
plt_log_mode(ax2)

num = len(pair_data.keys()) - 1
my_cm = lambda i: cm.coolwarm(i / num)

for (s1, s2), res_obj in results.items():
    lb = f"{s1}-{s2}"
    i = analysis.at[lb, "rank"]
    idx_trgt = int(analysis.at[lb, "idx"])
    # idx_trgt = np.searchsorted(wl_laser, 320.0)

    n_dist = utils.number_density_to_ppm(
        res_obj.res[:, idx_trgt] - cfs[(s1, s2)][:, idx_trgt], 
        res_obj.coord.z
    )
    stat_err_dist = utils.number_density_to_ppm(
        res_obj.stat_err[:, idx_trgt], 
        res_obj.coord.z
    )

    contam_err_wl = utils.number_density_to_ppm(
        res_obj.res[-1, :] - cfs[(s1, s2)][-1, :] - res_obj.n_true["SO2"][-1],
        res_obj.coord.z[-1],
    )
    stat_err_wl = utils.number_density_to_ppm(
        res_obj.stat_err[-1,:], 
        res_obj.coord.z[-1]
    )
    ax_dist.errorbar(
        x=res_obj.coord.distance,
        y=n_dist,
        yerr=stat_err_dist,
        capsize=8,
        fmt="o",
        markersize=6,
        ecolor=my_cm(i),
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )
    ax_err1_wl.plot(
        wl_laser,
        np.abs(contam_err_wl),
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )
    ax_err2_wl.plot(
        wl_laser,
        np.abs(stat_err_wl),
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )

    ax2.plot(
        lidar_coord.distance,
        debug[(s1, s2)].p_on[:, idx_trgt],
        ls="-",
        color=my_cm(i),
        zorder=num - i,
    )
    ax2.plot(
        lidar_coord.distance,
        debug[(s1, s2)].p_off[:, idx_trgt],
        ls="--",
        color=my_cm(i),
        zorder=num - i,
    )
    # ax2.scatter(
    #     lidar_coord.distance, debug[(s1, s2)].p_off[:, idx_trgt],
    #     marker="o", edgecolor=my_cm(i), facecolor="white", zorder=num-i
    # )

    pt_step = 5
    ax3.plot(
        res_obj.coord.distance,
        n_dist,
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )
    # ax3.errorbar(
    #     x=res_obj.coord.distance[::pt_step],
    #     y=n_dist[::pt_step],
    #     yerr=stat_err_dist[::pt_step],
    #     capsize=8,
    #     fmt="o",
    #     markersize=6,
    #     ecolor=my_cm(i),
    #     color=my_cm(i),
    #     label=f"{s1}-{s2}",
    #     zorder=num - i,
    # )

    ax3_ins1.plot(
        res_obj.coord.distance,
        n_dist,
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )
    # ax3_ins1.errorbar(
    #     x=res_obj.coord.distance,
    #     y=n_dist,
    #     yerr=stat_err_dist,
    #     capsize=8,
    #     fmt="o",
    #     markersize=6,
    #     ecolor=my_cm(i),
    #     color=my_cm(i),
    #     label=f"{s1}-{s2}",
    #     zorder=num - i,
    # )
    # ax3_ins2.errorbar(
    #     x=res_obj.coord.distance,
    #     y=utils.number_density_to_ppm(
    #         res_obj.res[:, idx_trgt] - cfs[(s1, s2)][:, idx_trgt], res_obj.coord.z
    #     ),
    #     yerr=utils.number_density_to_ppm(
    #         res_obj.stat_err[:, idx_trgt], res_obj.coord.z
    #     ),
    #     capsize=8,
    #     fmt="o",
    #     markersize=6,
    #     ecolor=my_cm(i),
    #     color=my_cm(i),
    #     label=f"{s1}-{s2}",
    #     zorder=num - i,
    # )


new_coord = lidar_coord.with_(
    distance=np.linspace(lidar_coord.distance[0], lidar_coord.distance[-1], 1000)
)
ax_dist.plot(
    new_coord.distance,
    env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"],
    c="black",
)
ax3.plot(
    new_coord.distance,
    env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"],
    c="black",
)
ax3_ins1.plot(
    new_coord.distance,
    env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"],
    c="black",
)
# ax3_ins2.plot(
#     new_coord.distance,
#     env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"],
#     c="black",
# )
x=np.logspace(0, 3, 1000)
for i, lb in enumerate(PSC.classification_label):
    y=field.spread.lateral(x, lb) 
    z=field.spread.vertical(x, lb)
    ax_pasq_l.plot(x, y, c=cm.viridis(i/6), label=PSC.classification_label_formatter(lb))
    ax_pasq_v.plot(x, z, c=cm.viridis(i/6), label=PSC.classification_label_formatter(lb))

color_handles = []
for lb, row in analysis.sort_values(by="rank")[["rank"]].iterrows():
    color_handles.append(
        Line2D([0], [0], color=my_cm(row["rank"]), ls="None", marker="o", label=lb)
    )
    
fig1_handles = color_handles.copy()
fig1_handles.insert(
    0,
    Line2D([0], [0], color="black", lw=2, ls="-", label=r"True"),
)
fig2_handles = color_handles.copy()
fig2_handles.append(Line2D([0], [0], linestyle="None", label=" "))
fig2_handles.extend(
    [
        Line2D([0], [0], color="black", lw=2, ls="-", label=r"$P_{\rm{on}}$"),
        Line2D([0], [0], color="black", lw=2, ls="--", label=r"$P_{\rm{off}}$"),
    ]
)

# fig2_handles.extend(
#     [
#         Line2D([0], [0], color="black", lw=2, ls="-", label=r"$P_{\rm{on}}$"),
#         Line2D([0], [0], color="black", lw=2, ls="--", label=r"$P_{\rm{off}}$"),
#     ]
# )

ax_pasq_l.set(
    xlabel="Downwind distance [m]",
    ylabel="Leteral spreadwidth [m]",
    xlim=(x.min(), x.max()),
    ylim=(1e-2, 1e4),
    xscale="log",
    yscale="log",
)
ax_pasq_v.set(
    xlabel="Downwind distance [m]",
    ylabel="Vertical spreadwidth [m]",
    xlim=(x.min(), x.max()),
    ylim=(1e-2, 1e4),
    xscale="log",
    yscale="log",
)
ax_dist.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"SO$_2$ Concentration [ppm]",
    xlim=(0, lidar_coord.distance[-1]),
)

ax_err1_wl.set_ylabel(r"Contamination error $\varepsilon$ [ppm]")
ax_err2_wl.set_ylabel(r"Statistical error $\Delta n$ [ppm]")
ax_err2_wl.set_xlabel("Laser wavelength [nm]")

ax2.set(
    xlabel="(Line of Sight) distance [m]", 
    ylabel=r"Received power $P_{phot}$"
)
ax3.set(
    xlabel="(Line of Sight) distance [m]",
    ylabel=r"SO$_2$ Concentration [ppm]",
    xlim=(0, lidar_coord.distance[-1]),
    ylim=(0, None),
)
ax3_ins1.set(xlim=[30, 70], ylim=[0, None])
# ax3_ins2.set(xlim=[300, 500], ylim=[0, 2])

ax_pasq_l.legend()
ax_pasq_v.legend()

ax_dist.legend(
    handles=fig1_handles, loc="upper right", frameon=True
)
ax2.legend(
    handles=fig2_handles,
    loc="upper right",
    frameon=True,
)
ax3.legend(
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    loc="upper left",
    handles=fig1_handles,
    frameon=True,
)

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(
    ax3,
    ax3_ins1,
    loc1=2,  # 左上
    loc2=4,  # 右下
    fc="none",
    ec="0.5"
)
# ax3.indicate_inset_zoom(ax3_ins2)
fig_pasq_l.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig_pasq_v.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig3.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

ext="pdf"
fig_pasq_l.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_04_field_spread_leteral." + ext,
)
fig_pasq_v.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_04_field_spread_vertical." + ext,
)
fig_env1.set_size_inches(3, 3)
fig_env2.set_size_inches(3, 3)
fig_env1.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_04_env_LoS." + ext,
)
fig_env2.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_04_env_image." + ext,
)
fig2.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_04_powers." + ext,
)
fig3.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_04_plume_meas." + ext,
)
plt.show(block=False)
input("PRESS ANY KEY...")
