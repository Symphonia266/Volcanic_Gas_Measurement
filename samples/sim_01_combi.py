# coding: utf-8
"""
NOTE:
- This is a **readability-only refactor** of the original script.
- **No logic, numerical behavior, plotting behavior, or side effects are changed.**
- Reordering is limited to grouping related code and extracting helper functions.
"""

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
import numpy as np
import pandas as pd
from matplotlib import cm, ticker, use
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter

# ================================
# Project-specific imports
# ================================
from gas_simulation import utils
from gas_simulation.consts import main_gases_props
from gas_simulation.atom import N, betas_N2, betas_O2
from gas_simulation.model import Gas, InstantEnvironment
from gas_simulation.lidar_model.utils import Coord
from gas_simulation.lidar_model import lidar, dial

# ================================
# Scripts config loading
# ================================
import config as cfg
plt.style.use(cfg.MPLSTYLE_PATH)
cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# Constants
# ================================
T_SEC = 60 * 10
AX_W = 2
AX_H = 4
THICK_FRONT = 300
THICK_BACK = 700


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


def plt_log_mode(ax, use_minor_grid=True, use_exponent=True):
    ax.set_yscale("log")
    # plt.grid(which='major',color='black',linestyle='-')
    # plt.grid(which='minor',color='lightgrey',linestyle='--', axis="y")
    if use_exponent:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain')
    if use_minor_grid:
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
lc = lidar.LidarCalc(M=100 * T_SEC)
dc = dial.DialCalc()
lidar_coord = Coord(
    distance=np.arange(lc.dR, 1000+lc.dR, lc.dR),
    theta_deg=0.0,
    x0=0,
    z0=1000,
)
lc.show_params()
dc.show_params()

# ================================
# Environment definition
# ================================
env = InstantEnvironment(
    gas={
        # 煙源直下で30ppmになるよう調整したQ
        # "SO2": Gas(Q=80e5, offset=0, cross_section=xs_SO2),
        # "H2S": Gas(Q=40e5, offset=0, cross_section=xs_H2S),
        # 今回の煙源位置、風プロファイル設定で30ppm程度になるよう調整したQ
        "SO2": Gas(Q=30, offset=0.07, cross_section=xs_SO2),
        "H2S": Gas(Q=15, offset=0.035, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
    time=T_SEC,
    trig=lambda x: ((x >= THICK_FRONT) & (x <= THICK_BACK)),
)
fig_env, ax_env = env.show_gases(lidar_coord)

# ================================
# Wavelength definitions
# ================================
wl_laser = np.arange(240, 370, 0.02)
wl = {
    "laser": wl_laser,
    "N2 Stokes": utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], False),
    "O2 Stokes": utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], False),
    "N2 a-Stokes": utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], True),
    "O2 a-Stokes": utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], True),
}

# ================================
# Backscatter and transmission
# ================================
beta_N2 = betas_N2(lidar_coord.z)
beta_O2 = betas_O2(lidar_coord.z)
tau = {k: env.transmittance(lidar_coord, v) for k, v in wl.items()}
beta_tau = {
    "N2 Stokes": beta_N2[:, np.newaxis] * tau["laser"] * tau["N2 Stokes"],
    "O2 Stokes": beta_O2[:, np.newaxis] * tau["laser"] * tau["O2 Stokes"],
    "N2 a-Stokes": beta_N2[:, np.newaxis] * tau["laser"] * tau["N2 a-Stokes"] * 0.1,
    "O2 a-Stokes": beta_O2[:, np.newaxis] * tau["laser"] * tau["O2 a-Stokes"] * 0.1,
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
        "stat_err_ppm@700",
        "stat_err_ppm@end",
        "contam_err_ppm@100",
        "contam_err_ppm@500",
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
    idx_100 = np.searchsorted(results[(s1, s2)].coord.distance, 100)-1
    idx_500 = np.searchsorted(results[(s1, s2)].coord.distance, 500)-1
    idx_700 = np.searchsorted(results[(s1, s2)].coord.distance, 700)-1
    idx_1000 =-1
    analysis.loc[lbs(s1, s2)] = pd.Series(
        {
            "idx": idx_wl_trgt,
            "wl_ls": wl_laser[idx_wl_trgt],
            "wl_on": results[(s1, s2)].wl_on[idx_wl_trgt],
            "wl_off": results[(s1, s2)].wl_off[idx_wl_trgt],
            "stat_err_ppm@700": utils.number_density_to_ppm(
                results[(s1, s2)].stat_err[idx_700, idx_wl_trgt],
                results[(s1, s2)].coord.z[idx_700],
            ),
            "stat_err_ppm@end": utils.number_density_to_ppm(
                results[(s1, s2)].stat_err[-1, idx_wl_trgt],
                results[(s1, s2)].coord.z[-1],
            ),
            "contam_err_ppm@100": utils.number_density_to_ppm(
                results[(s1, s2)].res[idx_100, idx_wl_trgt] - results[(s1, s2)].n_true["SO2"][idx_100],
                results[(s1, s2)].coord.z[idx_100],
            ),
            "contam_err_ppm@500": utils.number_density_to_ppm(
                results[(s1, s2)].res[idx_500, idx_wl_trgt] - results[(s1, s2)].n_true["SO2"][idx_500],
                results[(s1, s2)].coord.z[idx_500],
            ),
            "d_xs": debug[(s1, s2)].d_xs[idx_wl_trgt],
        }
    )
analysis["rank"] = np.argsort(np.argsort(analysis[["stat_err_ppm@end"]].values.flatten()))
print(analysis.sort_values(by="rank"))
# analysis.to_csv("samples/sim_result/anlysis.csv")

# ================================
# Result Printing (Raw text)
# ================================
# for (s1, s2), res_obj in results.items():
#     print(f"最遠方濃度{utils.number_density_to_ppm(res_obj.n_true["SO2"][-1], res_obj.coord.z[-1]):.3f} [ppm]")
#     lb=lbs(s1, s2)
#     i = analysis.at[lb, "rank"]
#     idx_trgt = int(analysis.at[lb, "idx"])
#     idx_trgt = np.searchsorted(wl_laser, 320.0)

#     cond = analysis.at[lb, "wl_on"] < analysis.at[lb, "wl_off"]
#     txt_wl_ls = f"{analysis.at[lb, "wl_ls"]:.2f}"
#     txt_on = f"on :{analysis.at[lb, "wl_on"]:.1f} nm"
#     txt_off = f"off:{analysis.at[lb, "wl_off"]:.1f} nm"
#     a:str = txt_on if cond else txt_off
#     b:str = txt_off if cond else txt_on
#     txt_staterr = f"{analysis.at[lb, 'stat_err_ppm']:.4f} [ppm]"
#     print(f"[{s1}-{s2}]target is {txt_wl_ls}({a}, {b}) [nm] : {txt_staterr}")


# ================================
# Plotting
# ================================
num = len(pair_data.keys()) - 1
my_cm = lambda i: cm.coolwarm(i / num)

# 受信光子数グラフエリア
# fig1,ax1 = fig_with_fixed_ax(ax_size=(AX_H, AX_W))
fig1, ax1 = plt.subplots()

# 測定シミュレーション結果グラフエリア
# fig2,ax2 = fig_with_fixed_ax(ax_size=(AX_H, AX_W), right=0.68)
fig2, ax2 = plt.subplots()
# ax2_insetをax2内に用意する
ins_top_space = 0.02
ins_right_space = 0.02
ins_width = 0.2
ins_height = 0.6
ax2_ins = ax2.inset_axes(
    [
        1 - ins_width - ins_right_space,
        1 - ins_height - ins_top_space,
        ins_width,
        ins_height,
    ]
)
# 統計誤差 / 干渉誤差グラフエリア
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()
fig8, ax8 = plt.subplots()
# ax8_ins = ax8.inset_axes([])
# fig3,ax3 = fig_with_fixed_ax(ax_size=(AX_H, AX_W), right=0.68)
# fig4,ax4 = fig_with_fixed_ax(ax_size=(AX_H, AX_W), right=0.68)
# fig5,ax5 = fig_with_fixed_ax(ax_size=(AX_H, AX_W), top=0.85)
# fig6,ax6 = fig_with_fixed_ax(ax_size=(AX_H, AX_W))
plt_log_mode(ax1, use_minor_grid=False)
plt_log_mode(ax3, use_minor_grid=False)
plt_log_mode(ax4, use_minor_grid=False)
# plt_log_mode(ax7)
plt_log_mode(ax8, use_minor_grid=False, use_exponent=False)
ax5.grid(False)
ax6.grid(False)
ax7.grid(False)
ax8.grid(False)


# SO2設定値プロット
new_coord = lidar_coord.with_(
    distance=np.linspace(lidar_coord.distance[0], lidar_coord.distance[-1], 1000)
)
ax2.plot(
    new_coord.distance,
    env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"],
    c="black",
    label="True",
)
for (s1, s2), res_obj in results.items():
    # print(f"最遠方濃度{utils.number_density_to_ppm(res_obj.n_true["SO2"][-1], res_obj.coord.z[-1]):.3f} [ppm]")
    lb = lbs(s1, s2)
    i = float(analysis.at[lb, "rank"])
    idx_trgt = int(analysis.at[lb, "idx"])
    # idx_trgt = np.searchsorted(wl_laser, 320.0)

    contam_error_wl = utils.number_density_to_ppm(
        res_obj.res[-1, :] - res_obj.n_true["SO2"][-1],
        res_obj.coord.z[-1],
    )
    stat_error_wl = utils.number_density_to_ppm(
        res_obj.stat_err[-1, :], 
        res_obj.coord.z[-1]
    )
    n_dist = utils.number_density_to_ppm(
        res_obj.res[:, idx_trgt],
        res_obj.coord.z,
    )
    contam_error_dist = utils.number_density_to_ppm(
        res_obj.res[:, idx_trgt] - res_obj.n_true["SO2"],
        res_obj.coord.z,
    )
    stat_error_dist = utils.number_density_to_ppm(
        res_obj.stat_err[:, idx_trgt], 
        res_obj.coord.z
    )
    if (s1, s2) == (
        "O2 Stokes", 
        "O2 a-Stokes"
    ) or (s1, s2) == (
        "N2 Stokes",
        "O2 Stokes",
    ):
        ax1.plot(
            lidar_coord.distance,
            debug[(s1, s2)].p_on[:, idx_trgt],
            ls="-",
            color=my_cm(i),
            zorder=num - i,
        )
        ax1.plot(
            lidar_coord.distance,
            debug[(s1, s2)].p_off[:, idx_trgt],
            ls="--",
            color=my_cm(i),
            zorder=num - i,
        )
    pt=10
    ax2.plot(        
        res_obj.coord.distance, 
        n_dist,
        ls="-",
        color=my_cm(i),
        zorder=num - i,
    )
    ax2.errorbar(
        x=res_obj.coord.distance[::pt],
        y=n_dist[::pt],
        yerr=stat_error_dist[::pt],
        ecolor=my_cm(i),
        color=my_cm(i),
        capsize=4,
        fmt="o",
        markersize=3,
        zorder=num - i,
        label=f"{s1:>13}-{s2:>13}",
    )
    ax7.plot(
        res_obj.coord.distance,
        n_dist,
        color=my_cm(i),
        zorder=num - i,
        label=f"{s1:>13}-{s2:>13}",
    )
    ax8.plot(
        res_obj.coord.distance,
        stat_error_dist,
        color=my_cm(i),
        zorder=num - i,
        label=f"{s1:>13}-{s2:>13}",
    )
    if (s1, s2) == ("O2 Stokes", "O2 a-Stokes") or (s1, s2) == (
        "N2 Stokes",
        "O2 Stokes",
    ):
        ax2_ins.errorbar(
            x=res_obj.coord.distance,
            y=n_dist,
            yerr=stat_error_dist,
            ecolor=my_cm(i),
            color=my_cm(i),
            capsize=4,
            fmt="o",
            markersize=3,
            zorder=num - i,
        )

    idx = ~np.isnan(np.abs(contam_error_wl))
    ax3.plot(
        wl_laser[idx],
        np.abs(contam_error_wl[idx]),
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )
    ax4.plot(
        wl_laser[idx],
        np.abs(stat_error_wl[idx]),
        color=my_cm(i),
        label=f"{s1}-{s2}",
        zorder=num - i,
    )

    spec = np.linspace(240, 370, 1000)
    if (s1, s2) == ("O2 Stokes", "O2 a-Stokes") or (s1, s2) == (
        "N2 Stokes",
        "O2 Stokes",
    ):
        ls = analysis.loc[lbs(s1, s2), ["wl_ls", "wl_on", "wl_off"]]
        # print(f"{s1}-{s2}"+r"$\sigma_{\rm{SO2}}$"+f":{xs_SO2(ls["wl_on"])-xs_SO2(ls["wl_off"])}"+r"[m$^2$]")
        # ax5.vlines(x=ls.values, ymin=[0, 0, 0], ymax=xs_SO2(ls.values), colors=my_cm(i))
        ax5.scatter(ls.values, xs_SO2(ls.values), c=my_cm(i), zorder=2, clip_on=False)
        
        if (s1, s2) == ("N2 Stokes","O2 Stokes"):      offset = 11e-23
        if (s1, s2) == ("O2 Stokes","O2 a-Stokes"):    offset = 9.2e-23
        ax5.text(
            x=ls["wl_ls"],
            y=offset,
            s=f"ls:{ls["wl_ls"]:.1f}", 
            ha="left", va="bottom", rotation=50, c=my_cm(i),
        )
        ax5.text(
            x=ls["wl_on"],
            y=offset,
            s=f"on:{ls["wl_on"]:.1f}",
            ha="left", va="bottom", rotation=50, c=my_cm(i),
        )
        ax5.text(
            x=ls["wl_off"],
            y=offset,
            s=f"off:{ls["wl_off"]:.1f}",
            ha="left", va="bottom", rotation=50, c=my_cm(i),
        )
        ax5.axvline(x=ls["wl_ls"], ls="-", c=my_cm(i), zorder=2)
        ax5.axvline(x=ls["wl_on"], ls="--", c=my_cm(i), zorder=2)
        ax5.axvline(x=ls["wl_off"], ls="--", c=my_cm(i), zorder=2)
        print(ls)
    # ax2.scatter(
    #     lidar_coord.distance, debug[(s1, s2)].p_off[:, idx_trgt],
    #     marker="o", edgecolor=cm_for_p_off(i), facecolor="white", zorder=num-i
    # )
ax5.plot(spec, xs_SO2(spec), c="black", zorder=1)

ax6.plot(spec, N(lidar_coord.z0) * 1e-6 * xs_O3(spec), c="black")
ax6.plot(spec, N(lidar_coord.z0) * 30e-6 * xs_SO2(spec), c="black")
ax6.plot(spec, N(lidar_coord.z0) * 15e-6 * xs_H2S(spec), c="black")

offset = 0.007
x, y = 250, 0.004
ax6.text(x=x, y=y+offset, va="center", ha="left", s=r"H$_2$S")
ax6.text(x=x, y=y, va="center", ha="left", s=r"(15ppm)")

x, y = 270, 0.02
ax6.text(x=x, y=y+offset, va="center", ha="left", s=r"O$_3$")
ax6.text(x=x, y=y, va="center", ha="left", s=r"(1ppm)")

x, y = 310, 0.03
ax6.text(x=x, y=y+offset, va="center", ha="left", s=r"SO$_2$")
ax6.text(x=x, y=y, va="center", ha="left", s=r"(30ppm)")

# ================================
# Generate color legends
# ================================
color_handles = []
for lb, row in analysis.sort_values(by="rank")[["rank"]].iterrows():
    color_handles.append(
        Line2D([0], [0], color=my_cm(row["rank"]), ls="None", marker="o", label=lb)
    )
# fig1_handles = color_handles.copy()
# fig1_handles.append(Line2D([0], [0], linestyle="None", label=" "))
# fig1_handles.extend(
#     [
#         Line2D([0], [0], color="black", lw=2, ls="-", label=r"$P_{\rm{on}}$"),
#         Line2D([0], [0], color="black", lw=2, ls="--", label=r"$P_{\rm{off}}$"),
#     ]
# )
fig1_handles = [
    Line2D([0], [0], color=my_cm(0.0), ls="None", marker="o", label=lbs("O2 Stokes", "O2 a-Stokes")),
    Line2D([0], [0], color=my_cm(6.0), ls="None", marker="o", label=lbs("O2 Stokes", "N2 Stokes")),
    Line2D([0], [0], color="black", ls="-", label=r"$P_{\rm{on}}$"),
    Line2D([0], [0], color="black", ls="--", label=r"$P_{\rm{off}}$")
]
fig2_handles = color_handles.copy()
fig7_handles = color_handles.copy()
fig7_handles.insert(0, Line2D([0], [0], linestyle="-", c="black", label=r"$\rm{SO_2}$ setting"))
# fig2_handles.insert(
#     0,
#     Line2D([0], [0], color="black", lw=2, ls="-", label=r"True"),
# )

ax2.indicate_inset_zoom(ax2_ins)

ax_env.set_ylim(-5, 40)
ax1.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"Received power $P_{\rm{phot}}$",
    xlim=(0, lidar_coord.distance.max()),
)
ax2.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"Concentrations of SO$_2$ [ppm]",
    xlim=(200, lidar_coord.distance.max()),
)
ax2_ins.set(xlim=(670, 700), ylim=(25, 31))
ax3.set(
    xlabel=r"Laser wavelength [nm]",
    ylabel=r"Contamination errors [ppm]",
    xlim=(240, None)
)
ax4.set(
    xlabel=r"Laser wavelength [nm]",
    ylabel=r"Statistical errors [ppm]",
    xlim=(240, None)
)
ax5.set(
    xlabel=r"Wavelength [nm]",
    # ylabel=r"Absorption cross section |$\sigma_{\rm{SO2}}$| [m$^{2}$]",
    ylabel=r"Absorption cross section [m$^{2}$]",
    xlim=[None, 370],
    ylim=[0, 9e-23],
)
ax6.set(
    xlabel=r"Wavelength [nm]",
    ylabel=r"Extinction coefficient [m$^{-1}$]",
    xlim=[240, 370],
    ylim=[0, 0.07],
)
ax7.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"Contamination error [ppm]",
    xlim=(0, lidar_coord.distance.max()),
    # xlim=(300, 700),
    # ylim=(25, 31),
)
ax8.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"Statistical error [ppm]",
    xlim=(0, lidar_coord.distance.max()),
    # ylim=(1e-1, 1e1),
)

# ax_env.legend(
#     bbox_to_anchor=(1.02, 1.0),
#     borderaxespad=0,
#     loc="upper left",
#     frameon=True,
# )
# ax1.legend(
#     bbox_to_anchor=(0.98,0.98),
#     borderaxespad=0,
#     loc="upper right",
#     handles=fig1_handles,
#     frameon=True,
# )
# ax2.legend(
#     bbox_to_anchor=(1.02, 1),
#     borderaxespad=0,
#     loc="upper left",
#     handles=fig2_handles,
#     frameon=True,
# )
# ax3.legend(
#     bbox_to_anchor=(1.02, 1),
#     borderaxespad=0,
#     loc="upper left",
#     handles=color_handles,
#     frameon=True,
# )
# ax4.legend(
#     bbox_to_anchor=(1.02, 1),
#     borderaxespad=0,
#     loc="upper left",
#     handles=color_handles,
#     frameon=True,
# )
# ax7.legend(
#     bbox_to_anchor=(1.02, 1),
#     borderaxespad=0,
#     loc="upper left",
#     handles=fig2_handles,
#     frameon=True,
# )
# ax7.legend(
#     bbox_to_anchor=(1.02, 1),
#     borderaxespad=0,
#     loc="upper left",
#     handles=fig2_handles,
#     frameon=True,
# )
# ax8.legend(
#     bbox_to_anchor=(1.02, 1),
#     borderaxespad=0,
#     loc="upper left",
#     handles=fig2_handles,
#     frameon=True,
# )

# ================================
# Final output
# ================================

ax_env.xaxis.set_major_locator(ticker.LinearLocator(6))
# ax_env.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# ax1.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
# ax2.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
# ax3.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
# ax4.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
ax5.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax5.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax5.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

fig_env.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig1.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig3.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig4.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig5.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig6.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig7.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig8.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

fig_env.savefig(str(cfg.OUT_DIR / f"sim_01_env_LoS.{cfg.EXT}"), format=cfg.EXT)
# fig2.savefig(str(cfg.OUT_DIR / f"sim_01_n_SO2_6combies.{cfg.EXT}"), format=cfg.EXT)
fig1.savefig(str(cfg.OUT_DIR / f"sim_01_power.{cfg.EXT}"), format=cfg.EXT)
fig3.savefig(str(cfg.OUT_DIR / f"sim_01_contami_error_6combies.{cfg.EXT}"), format=cfg.EXT)
fig4.savefig(str(cfg.OUT_DIR / f"sim_01_stat_error_6combies.{cfg.EXT}"), format=cfg.EXT)
fig5.savefig(str(cfg.OUT_DIR / f"sim_01_cond_2case.{cfg.EXT}"), format=cfg.EXT)
fig6.savefig(str(cfg.OUT_DIR / f"sim_01_3gases_absorp.{cfg.EXT}"), format=cfg.EXT)
fig7.savefig(str(cfg.OUT_DIR / f"sim_01_6combies_cntm.{cfg.EXT}"), format=cfg.EXT)
fig8.savefig(str(cfg.OUT_DIR / f"sim_01_6combies_stat.{cfg.EXT}"), format=cfg.EXT)

# plt.show(block=False)
# input("PRESS ANY KEY...")
