# coding: utf-8
import sys
from turtle import distance
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from matplotlib import cm, legend
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view as np_SWV

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

plt.rcParams['font.family'] ='sans-serif'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 8 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.locator_params(axis='y',nbins=6)#y軸，6個以内．

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

t_sec = 60 * 30  # [sec]
lc = lidar.LidarCalc(M=100 * t_sec)
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
        "SO2": Gas(Q=30, offset=0.07, cross_section=xs_SO2),
        "H2S": Gas(Q=15, offset=0.035, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
    time=t_sec,
    trig=lambda x: ((x >= 300) & (x <= 700)),
)
env.show_gases(lidar_coord)

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
tau = {k:env.transmittance(lidar_coord, v) for k, v in wl.items()}
beta_tau = {
    "N2_st" : beta_N2[:, np.newaxis]*tau["laser"]*tau["N2_st"], 
    "O2_st" : beta_O2[:, np.newaxis]*tau["laser"]*tau["O2_st"], 
    "N2_as" : beta_N2[:, np.newaxis]*tau["laser"]*tau["N2_as"]*0.1, 
    "O2_as" : beta_O2[:, np.newaxis]*tau["laser"]*tau["O2_as"]*0.1
}

# === power calculation ===
p = {
    k:lc.power(
        dist=lidar_coord.distance[:, np.newaxis],
        wl=wl_laser[np.newaxis, :],
        beta_tau=v,
    )
    for k, v, in beta_tau.items()
}
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

results:dict[tuple[str, str], dial.DialResult] = {}
cfs:dict[tuple[str, str], np.ndarray] = {}
debug:dict[tuple[str, str], dial.DialDebugData] = {}
analysis:pd.DataFrame = pd.DataFrame([], columns=[
    "idx", "wl_ls", "wl_on", "wl_off", "stat_err_ppm", "contam_err_ppm", "d_xs", 
])

for (s1, s2), dial_input in pair_data.items():
    print(f"calculation ({s1}, {s2}) combination ramanDIAL...")

    # === DIAL calculation ===
    results[(s1, s2)], cf_input, debug[(s1, s2)]= dc.estimate(dial_input)
    cfs[(s1, s2)] = dial.calc_correction_factor(cf_input, mol=True, aer=True)
    idx_wl_trgt = np.nanargmin(np.abs(results[(s1, s2)].stat_err[-1,:]))

    analysis.loc[f"{s1}-{s2}"] = pd.Series({
        "idx" : idx_wl_trgt,
        "wl_ls" : wl_laser[idx_wl_trgt],
        "wl_on" : results[(s1, s2)].wl_on[idx_wl_trgt],
        "wl_off" : results[(s1, s2)].wl_off[idx_wl_trgt],
        "stat_err_ppm" : utils.number_density_to_ppm(
            results[(s1, s2)].stat_err[-1, idx_wl_trgt], 
            results[(s1, s2)].coord.z[-1]
        ),
        "contam_err_ppm": utils.number_density_to_ppm(
            results[(s1, s2)].res[-1, idx_wl_trgt]-cfs[(s1, s2)][-1, idx_wl_trgt], 
            results[(s1, s2)].coord.z[-1]
        ),
        "d_xs" : debug[(s1, s2)].d_xs[idx_wl_trgt],
    })

analysis["rank"] = np.argsort(np.argsort(analysis[["stat_err_ppm"]].values.flatten()))
print(analysis.sort_values(by="rank"))


num = len(pair_data.keys())-1
cm_for_combinations = lambda i: cm.coolwarm(i/num)
cm_for_p_on = lambda i: cm.coolwarm(i/num)
cm_for_p_off = lambda i: cm.coolwarm(i/num)

# 受信光子数グラフエリア
fig1,ax1 = plt.subplots(1,1,figsize=(3.14,3.14))
ax1.grid(which="major", ls="-", c="darkgrey")
ax1.grid(which="minor", ls="--", c="lightgrey")
ax1.set_axisbelow(True)
ax1.set_yscale("log")
ax1.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"Received power $P_{\rm{phot}}$",
    xlim=(0, lidar_coord.distance.max())
)

# 測定シミュレーション結果グラフエリア
fig2,ax2 = plt.subplots(1,1,figsize=(3.14,3.14))
ax2.grid(which="major", ls="-", c="darkgrey")
ax2.grid(which="minor", ls="--", c="lightgrey")
ax2.set_axisbelow(True)
ax2.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"Concentrations of SO$_2$ [ppm]",
    xlim=(0, lidar_coord.distance.max())
)

# 統計誤差 / 干渉誤差グラフエリア
fig3, axes3 = plt.subplots(2,1,figsize=(3.14,3.14))
for ax in axes3:
    ax.grid(which="major", ls="-", c="darkgrey")
    ax.grid(which="minor", ls="--", c="lightgrey")
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    
    
axes3[0].set(
    # xlabel= r"laser wavelength [nm]",
    ylabel=r"Contamination errors |$\varepsilon_{SO2}$| [ppm]",
    # xlim=(0, lidar_coord.distance.max())
)
axes3[1].set(
    xlabel= r"laser wavelength [nm]",
    ylabel=r"Statistical errors |s_{SO2}| [ppm] [ppm]",
)

# SO2設定値プロット
new_coord = lidar_coord.with_(
    distance=np.linspace(lidar_coord.distance[0], lidar_coord.distance[-1], 1000)
)
ax2.plot(
    new_coord.distance, 
    env.number_density_at(new_coord.x, 0, new_coord.z, ppm=True)["SO2"], 
    c="black"
)

for (s1, s2), res_obj in results.items():
    # print(f"最遠方濃度{utils.number_density_to_ppm(res_obj.n_true["SO2"][-1], res_obj.coord.z[-1]):.3f} [ppm]")
    lb = f"{s1}-{s2}"
    idx_trgt = int(analysis.at[lb, "idx"])
    # idx_trgt = np.searchsorted(wl_laser, 320.0)
    cond = analysis.at[lb, "wl_on"] < analysis.at[lb, "wl_off"]

    txt_wl_ls = f"{analysis.at[lb, "wl_ls"]:.2f}"
    txt_on = f"on :{analysis.at[lb, "wl_on"]:.1f} nm"
    txt_off = f"off:{analysis.at[lb, "wl_off"]:.1f} nm"
    a:str = txt_on if cond else txt_off
    b:str = txt_off if cond else txt_on
    txt_staterr = f"{analysis.at[lb, 'stat_err_ppm']:.4f} [ppm]"

    i = analysis.at[lb, "rank"]
    # print(f"[{s1}-{s2}]target is {txt_wl_ls}({a}, {b}) [nm] : {txt_staterr}")


    ax1.plot(
        lidar_coord.distance, debug[(s1, s2)].p_on[:, idx_trgt], 
        ls="-", color=cm_for_p_on(i), zorder=num-i
    )
    ax1.plot(
        lidar_coord.distance, debug[(s1, s2)].p_off[:, idx_trgt],
        ls="--", color=cm_for_p_off(i), zorder=num-i
    )
    
    ax2.errorbar(
        x=res_obj.coord.distance, 
        y=utils.number_density_to_ppm(
            res_obj.res[:, idx_trgt]-cfs[(s1, s2)][:, idx_trgt], 
            res_obj.coord.z
        ), 
        yerr=utils.number_density_to_ppm(
            res_obj.stat_err[:, idx_trgt], 
            res_obj.coord.z
        ),
        capsize=8, fmt='o', markersize=6, ecolor=cm_for_combinations(i), 
        color=cm_for_combinations(i), label=f"{s1}-{s2}", zorder=num-i
    )
    axes3[0].scatter(
        wl_laser, 
        utils.number_density_to_ppm(
            np.abs(res_obj.res[-1,:]-cfs[(s1, s2)][-1, :]-res_obj.n_true["SO2"][-1]), 
            res_obj.coord.z[-1]
        ),         
        color=cm_for_combinations(i), label=f"{s1}-{s2}", s=8, zorder=num-i
    )
    axes3[1].scatter(
        wl_laser, 
        utils.number_density_to_ppm(
            res_obj.stat_err[-1, :], 
            res_obj.coord.z[-1]
        ), 
        color=cm_for_combinations(i), label=f"{s1}-{s2}", s=3, zorder=num-i
    )
    # ax2.scatter(
    #     lidar_coord.distance, debug[(s1, s2)].p_off[:, idx_trgt],
    #     marker="o", edgecolor=cm_for_p_off(i), facecolor="white", zorder=num-i
    # )

color_handles = []
analysis_sorted = analysis.sort_values(by="rank")[["rank"]]
for lb, row in analysis_sorted.iterrows():
    color_handles.append(
        Line2D(
            [0], [0],
            color=cm_for_combinations(row["rank"]),
            ls="None",
            marker="o",
            label=lb
        )
    )
fig1_handles = color_handles.copy()
fig1_handles.append(
    Line2D([0], [0], linestyle="None", label=" ")
)
fig1_handles.extend([
    Line2D([0], [0], color="black", lw=2, ls="-",  label=r"$P_{\rm{on}}$"),
    Line2D([0], [0], color="black", lw=2, ls="--", label=r"$P_{\rm{off}}$"),
])

ax1.legend(
    loc="upper right",
    handles=fig1_handles,
    frameon=True,
)
ax2.legend(
    bbox_to_anchor=(1,1), 
    borderaxespad=0,
    loc='upper left',
    handles=color_handles,
    frameon=True,
)

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.show(block=False)
input("PRESS ANY KEY...")
