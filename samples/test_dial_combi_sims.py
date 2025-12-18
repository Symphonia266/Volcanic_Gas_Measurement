# coding: utf-8
import sys
from turtle import distance
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from matplotlib import pyplot as plt
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

t_sec = 60 * 30  # [sec]
lc = lidar.LidarCalc(dR=30, M=100 * t_sec)
dc = dial.DialCalc()

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
    trig=lambda x: ((x >= 400) & (x <= 600)),
)
env.show_gases(lidar_coord)

wl_laser = np.arange(240, 370, 0.02)
idx_320nm = np.searchsorted(wl_laser, 320)
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
    (s1, s2): {
        "p_s1": p[s1],
        "p_s2": p[s2],
        "wl_s1": wl[s1],
        "wl_s2": wl[s2],
    }
    for s1, s2 in combinations(p.keys(), 2)
}

@dataclass
class DialResult:
    wl_on               :np.ndarray
    wl_off              :np.ndarray
    coord               :Coord
    n_true              :np.ndarray
    res                 :np.ndarray
    stat_err            :np.ndarray
    correction_factor   :np.ndarray

results = {}
for (s1, s2), d in pair_data.items():
    print(f"calculation ({s1}, {s2}) combination ramanDIAL...")
    wl_on, wl_off, d_xs_SO2, p_on, p_off, _ = dial.onoff_swapper(
        wl_s1=d["wl_s1"], 
        wl_s2=d["wl_s2"], 
        xs_s1=env.gas_inventory["SO2"].cross_section(d["wl_s1"]), 
        xs_s2=env.gas_inventory["SO2"].cross_section(d["wl_s2"]), 
        p_s1 = d["p_s1"], 
        p_s2 = d["p_s2"]
    )

    # === DIAL calculation ===
    n_gas = env.number_density_at(lidar_coord.x, 0, lidar_coord.z)
    p_on_R1, p_on_R2, p_off_R1, p_off_R2, dial_dR, dial_coord, n_true = dial.prepare_diff(
        p_on, p_off, lidar_coord, n_gas
    )
    res = dc.calc(
        p_on_R1=p_on_R1,
        p_on_R2=p_on_R2,
        p_off_R1=p_off_R1,
        p_off_R2=p_off_R2,
        dR=dial_dR[:, np.newaxis],
        d_xs=d_xs_SO2[np.newaxis, :],
    )
    stat_err = dc.stat_error(
        p_on_R1=p_on_R1,
        p_on_R2=p_on_R2,
        p_off_R1=p_off_R1,
        p_off_R2=p_off_R2,
        dR=dial_dR[:, np.newaxis],
        d_xs=d_xs_SO2[np.newaxis, :],
    )
    correction_factor = dial.calc_dial_correction_factor(
        env=env,
        alt=dial_coord.z[:, np.newaxis],
        wl_on=wl_on[np.newaxis, :],
        wl_off=wl_off[np.newaxis, :],
        d_xs=d_xs_SO2[np.newaxis, :],
    )
    results[(s1, s2)] = DialResult(
        wl_on=wl_on,
        wl_off=wl_off,
        coord=dial_coord,
        n_true=n_true["SO2"],
        res=res, 
        stat_err=stat_err, 
        correction_factor=correction_factor
    )

coord = next(iter(results.values())).coord
n_true = next(iter(results.values())).n_true

new_coord = lidar_coord.with_(
    distance=np.linspace(lidar_coord.distance[0], lidar_coord.distance[-1], 1000)
)
n = env.number_density_at(new_coord.x, 0, new_coord.z)["SO2"]

fig, axes = plt.subplots(1,2)
axes[0].scatter(
    coord.distance, 
    utils.number_density_to_ppm(n_true, coord.z), 
    c="black", marker="*"
)
axes[0].plot(
    new_coord.distance, 
    utils.number_density_to_ppm(n, new_coord.z), 
    c="darkgrey"
)
axes[0].set(
    xlabel="(Line of Sight) distance [m]",
    # xlim = (r.min(), r.max()),
    ylabel=r"SO$_2$ concentration [ppm]",
)

plt.show(block=False)
input("PRESS ANY KEY...")
