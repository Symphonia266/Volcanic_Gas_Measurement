# coding: utf-8
from doctest import debug
import sys
from turtle import distance
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from matplotlib import cm, legend, markers
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view as np_SWV
from pytest import mark

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from gas_simulation import utils
from gas_simulation.consts import main_gases_props
from gas_simulation.atom import betas_N2, betas_O2

from gas_simulation.model import Gas, InstantEnvironment
from gas_simulation.lidar_model.utils import Coord
from gas_simulation.lidar_model import lidar
from gas_simulation.lidar_model import dial
from gas_simulation import result_viewer as viewer

plt.style.use("forThesis.mplstyle")

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

# print(xs_SO2(230.9))
# print(xs_H2S(230.9))
# print(xs_O3(230.9))

T_SEC = 60 * 10  # [sec]
lc = lidar.LidarCalc(M=100 * T_SEC)
dc = dial.DialCalc()
lc.show_params()
dc.show_params()

lidar_coord = Coord(
    distance=np.arange(lc.dR, 1000, lc.dR),
    theta_deg=0.0,
    x0=0,
    z0=1000,
)

env = InstantEnvironment(
    gas={
        # 煙源直下で30ppmになるよう調整したQ
        # "SO2": Gas(Q=80e5, offset=0, cross_section=xs_SO2),
        # "H2S": Gas(Q=40e5, offset=0, cross_section=xs_H2S),
        # 今回の煙源位置、風プロファイル設定で30ppm程度になるよう調整したQ
        "SO2": Gas(Q=100, offset=0.07, cross_section=xs_SO2),
        "H2S": Gas(Q=15, offset=0.035, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
    time=T_SEC,
    trig=lambda x: ((x >= 300) & (x <= 700)),
)
fig_env, ax_env = env.show_gases(lidar_coord)

wl_laser = np.arange(240, 370, 0.02)
wl = {
    "laser": wl_laser,
    "N2_st": utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], False),
    "O2_st": utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], False),
    "N2_as": utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], True),
    "O2_as": utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], True),
}

beta_N2 = betas_N2(lidar_coord.z)
beta_O2 = betas_O2(lidar_coord.z)
tau = {k: env.transmittance(lidar_coord, v) for k, v in wl.items()}
beta_tau = {
    "N2_st": beta_N2[:, np.newaxis] * tau["laser"] * tau["N2_st"],
    "O2_st": beta_O2[:, np.newaxis] * tau["laser"] * tau["O2_st"],
    "N2_as": beta_N2[:, np.newaxis] * tau["laser"] * tau["N2_as"] * 0.1,
    "O2_as": beta_O2[:, np.newaxis] * tau["laser"] * tau["O2_as"] * 0.1,
}

# === power calculation ===
p = {
    k: lc.power(
        dist=lidar_coord.distance[:, np.newaxis],
        wl=wl_laser[np.newaxis, :],
        beta_tau=v,
    )
    for k, v, in beta_tau.items()
}
obj_1 = dial.RamanShiftObject(
    wl=wl["O2_st"],
    xs=env.gas_inventory["SO2"].cross_section(wl["O2_st"]),
    p=p["O2_st"],
)
obj_2 = dial.RamanShiftObject(
    wl=wl["O2_as"],
    xs=env.gas_inventory["SO2"].cross_section(wl["O2_as"]),
    p=p["O2_as"],
)

res_lb = ["single", "sum5"]
num = len(res_lb) - 1

input_1 = dial.DialInput(obj_s1=obj_1, obj_s2=obj_2, lidar_coord=lidar_coord, env=env)
input_2 = input_1.with_(sumN=5)

res1, cf_input1, debug1 = dc.estimate(input_1)
res2, cf_input2, debug2 = dc.estimate(input_2)
cf1 = dial.calc_correction_factor(cf_input1)
cf2 = dial.calc_correction_factor(cf_input2)

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

# idx_trgt = np.nanargmin(res1.stat_err[-1, :])
idx_wl_trgt = np.searchsorted(wl_laser, 334.58)
# idx_dist_trgt = np.searchsorted(res1.coord.distance, 697)-1
idx_dist_trgt1 = np.searchsorted(res1.coord.distance, 700)-1
idx_dist_trgt2 = np.searchsorted(res2.coord.distance, 700)-1
# idx_dist_trgt = -1
print(f"{res1.coord.distance[idx_dist_trgt1]:.2f} m selected as target distance for single.")
print(f"{res2.coord.distance[idx_dist_trgt2]:.2f} m selected as target distance for sumN.")

for lb, res_obj, res_cf, debug_obj in zip(res_lb, [res1, res2], [cf1, cf2], [debug1, debug2]):
    idx_dist_trgt = np.searchsorted(res_obj.coord.distance, 700)-1
    analysis.loc[lb] = pd.Series(
        {
            "idx": idx_wl_trgt,
            "wl_ls": wl_laser[idx_wl_trgt],
            "wl_on": res_obj.wl_on[idx_wl_trgt],
            "wl_off": res_obj.wl_off[idx_wl_trgt],
            "stat_err_ppm": utils.number_density_to_ppm(
                res_obj.stat_err[idx_dist_trgt, idx_wl_trgt], res_obj.coord.z[-1]
            ),
            "contam_err_ppm": utils.number_density_to_ppm(
                res_obj.res[idx_dist_trgt, idx_wl_trgt] - res_cf[idx_dist_trgt, idx_wl_trgt],
                res_obj.coord.z[idx_dist_trgt],
            ),
            "d_xs": debug_obj.d_xs[idx_wl_trgt],
        }
    )
print(analysis)
print(res1.coord.distance[idx_dist_trgt1])
print(res2.coord.distance[idx_dist_trgt2])
new_coord = lidar_coord.with_(
    distance=np.linspace(
        lidar_coord.distance[0], 
        lidar_coord.distance[-1],
        1000
    )
)
pt_step = 5
x1 = res1.coord.distance
y1_1 = utils.number_density_to_ppm(
    res1.res[:, idx_wl_trgt] - cf1[:, idx_wl_trgt], 
    res1.coord.z
)
y1_2 = utils.number_density_to_ppm(
    res1.stat_err[:, idx_wl_trgt], 
    res1.coord.z
)

x2 = res2.coord.distance
y2_1 = utils.number_density_to_ppm(
    res2.res[:, idx_wl_trgt], 
    res2.coord.z
)
y2_2 = utils.number_density_to_ppm(
    res2.stat_err[:, idx_wl_trgt], 
    res2.coord.z
)


fig1 = plt.figure()
fig1.tight_layout(pad=0.4)
fig1.subplots_adjust(
    left=0.08,    # ← 左余白（小さくすると詰まる）
    right=0.86,   # ← 右余白（legend 領域込み）
    wspace=0.15   # ← 左右 axes 間隔（効いている）
)
gs = GridSpec(
    nrows=2,
    ncols=2,
    figure=fig1,
    width_ratios=[3.7, 2],
    height_ratios=[3, 1],
    left=0.08,
    right=0.96,   # ← ここが重要（右切れ防止）
    bottom=0.12,
    top=0.95,
    wspace=0.16,
    hspace=0.15,
)

ax1 = fig1.add_subplot(gs[:, 0])      # main plot
ax1_ins = fig1.add_subplot(gs[0, 1])  # zoom (inset-like)
ax1_leg = fig1.add_subplot(gs[1, 1])  # legend area
ax1_leg.axis("off")

fig2, ax2 = plt.subplots()
ax1.grid(False)
ax1_ins.grid(False)
ax1_leg.grid(False)
ax2.grid(False)

def plotter(x1, y1_1, y1_2, x2, y2_1, y2_2, pt_step, ax):
    ax.plot(
        new_coord.distance,
        env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"],
        c="black",
        ls="--",
        label="SO2 setting",
    )
    ax.plot(x1, y1_1, c=cm.viridis(0.9))
    ax.errorbar(
        x=x1[::pt_step],
        y=y1_1[::pt_step],
        yerr=y1_2[::pt_step],
        capsize=8,
        fmt="o",
        markersize=6,
        zorder=1,
        ecolor=cm.viridis(0.9),
        color=cm.viridis(0.9),
        label=r"$\Delta R$=5 m",
    )
    ax.plot(x2, y2_1, c=cm.viridis(0.1))
    ax.errorbar(
        x=x2[::pt_step],
        y=y2_1[::pt_step],
        yerr=y2_2[::pt_step],
        capsize=8,
        fmt="o",
        markersize=6,
        zorder=2,
        ecolor=cm.viridis(0.1),
        color=cm.viridis(0.1),
        label=r"$\Delta R$= 25 m",
    )
plotter(x1, y1_1, y1_2, x2, y2_1, y2_2, pt_step, ax1)
plotter(x1, y1_1, y1_2, x2, y2_1, y2_2, pt_step, ax1_ins)

ax2.plot(
    debug1.coord.distance,
    debug1.p_on[:, idx_wl_trgt],
    # marker="o",
    label="single",
    color="darkgrey",
)
ax2.plot(
    debug2.coord.distance,
    debug2.p_on[:, idx_wl_trgt],
    # marker="^",
    label="5 points summation for the front and rear",
    color="black",
)

ax_env.set_ylim(0, 110)
ax1.set(
    xlabel="(Line of Sight) distance [m]",
    ylabel="concentration [ppm]",
    xlim=(0, lidar_coord.distance.max()),
)
ax1_ins.set(
    xlim=(670, 730),
)
ax2.set(
    xlabel="(Line of Sight) distance [m]",
    ylabel=r"received power $P_{\rm{on}}$",
    xlim=(0, lidar_coord.distance.max()),
    yscale="log",
)

handles, labels = ax1.get_legend_handles_labels()
ax1_leg.legend(
    handles,
    labels,
    loc="center right",   # ← 右側に寄せる
    bbox_to_anchor=(1.0, 0.01),  # ← 下にずらす（0.0 に近いほど下）
    frameon=True,
)
ax2.legend(loc="upper left")
# fig1.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

ext="pdf"
fig_env.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_02_1_env." + ext,
)
fig1.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_02_dR-5mto25m." + ext,
)
fig2.savefig(
    format=ext,
    dpi=400,
    bbox_inches="tight",
    fname="samples/sim_result/sim_02_power." + ext,
)

plt.show(block=False)
input("PRESS ANY KEY...")
