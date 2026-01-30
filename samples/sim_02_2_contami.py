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

plt.style.use("my_sty.mplstyle")

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
        "SO2": Gas(Q=30, offset=0.07, cross_section=xs_SO2),
        "H2S": Gas(Q=1000, offset=0.035, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
    time=T_SEC,
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

input_1 = dial.DialInput(
    env=env,
    obj_s1=obj_1,
    obj_s2=obj_2,
    lidar_coord=lidar_coord,
)

res1, cf_input1, debug1 = dc.estimate(input_1)

idx_wl_trgt = np.nanargmin(np.abs(res1.stat_err[-1,:]))
idx_dist_trgt = np.searchsorted(res1.coord.distance, 500)

print(f"{wl_laser[idx_wl_trgt]:.2f} nm selected as target wavelength.")
print(f"{res1.coord.distance[idx_dist_trgt]:.2f} m selected as target distance.")

temp = utils.number_density_to_ppm(
  res1.res[idx_dist_trgt, idx_wl_trgt], 
  res1.coord.z[idx_dist_trgt]
)
print(f"{temp:.2f} ppm estimated at target point.")

est = [500, 600, 800]  # [ppm]
est_logic = lambda x: (x >= 500-100) & (x <= 500+100)
n_O3_est = np.full_like(res1.coord.distance, 0.005)
n_H2S_est = np.where(est_logic(res1.coord.distance), est[0], 0.0)

cf_0 = dial.calc_correction_factor(
  cf_input1,
  mol=True,aer=True,n_gas_est={
    "O3" : np.full_like(res1.coord.distance, 0.005)
})

cf_1 = dial.calc_correction_factor(
  cf_input1,
  mol=True,aer=True,n_gas_est={
    "H2S": np.where(est_logic(res1.coord.distance), est[0], 0.0),
    "O3" : np.full_like(res1.coord.distance, 0.005)
})
cf_2 = dial.calc_correction_factor(
  cf_input1,
  mol=True,aer=True,n_gas_est={
    "H2S": np.where(est_logic(res1.coord.distance), est[1], 0.0),
    "O3" : np.full_like(res1.coord.distance, 0.005)
})
cf_3 = dial.calc_correction_factor(
  cf_input1,
  mol=True,aer=True,n_gas_est={
    "H2S": np.where(est_logic(res1.coord.distance), est[2], 0.0),
    "O3" : np.full_like(res1.coord.distance, 0.005),
})

n_cf_ignore = utils.number_density_to_ppm(
  res1.res[:, idx_wl_trgt], 
  res1.coord.z
)
n_cf_0 = utils.number_density_to_ppm(
  res1.res[:, idx_wl_trgt] - cf_0[:, idx_wl_trgt], 
  res1.coord.z
)
n_cf_1 = utils.number_density_to_ppm(
  res1.res[:, idx_wl_trgt] - cf_1[:, idx_wl_trgt], 
  res1.coord.z
)
n_cf_2 = utils.number_density_to_ppm(
  res1.res[:, idx_wl_trgt] - cf_2[:, idx_wl_trgt], 
  res1.coord.z
)
n_cf_3 = utils.number_density_to_ppm(
  res1.res[:, idx_wl_trgt] - cf_3[:, idx_wl_trgt], 
  res1.coord.z
)
eps = np.array([
    n_cf_ignore-utils.number_density_to_ppm(res1.n_true["SO2"], res1.coord.z),
    n_cf_0-utils.number_density_to_ppm(res1.n_true["SO2"], res1.coord.z),
    n_cf_1-utils.number_density_to_ppm(res1.n_true["SO2"], res1.coord.z),
    n_cf_2-utils.number_density_to_ppm(res1.n_true["SO2"], res1.coord.z),
    n_cf_3-utils.number_density_to_ppm(res1.n_true["SO2"], res1.coord.z)
])
# print(eps[0])
print(f"est {0:<8.4g} [ppm] : {n_cf_ignore[0]:< 10.5g}({eps[0,0]:< 10.5g}), {n_cf_ignore[idx_dist_trgt]:< 10.5g}({eps[0,idx_dist_trgt]:< 10.5g}) [ppm]")
print(f"est {0:<8.4g} [ppm] : {n_cf_0[0]:< 10.5g}({eps[1,0]:< 10.5g}), {n_cf_0[idx_dist_trgt]:< 10.5g}({eps[1,idx_dist_trgt]:< 10.5g}) [ppm]")
print(f"est {est[0]:<8.4g} [ppm] : {n_cf_1[0]:< 10.5g}({eps[2,0]:< 10.5g}), {n_cf_1[idx_dist_trgt]:< 10.5g}({eps[2,idx_dist_trgt]:< 10.5g}) [ppm]")
print(f"est {est[1]:<8.4g} [ppm] : {n_cf_2[0]:< 10.5g}({eps[3,0]:< 10.5g}), {n_cf_2[idx_dist_trgt]:< 10.5g}({eps[3,idx_dist_trgt]:< 10.5g}) [ppm]")
print(f"est {est[2]:<8.4g} [ppm] : {n_cf_3[0]:< 10.5g}({eps[4,0]:< 10.5g}), {n_cf_3[idx_dist_trgt]:< 10.5g}({eps[4,idx_dist_trgt]:< 10.5g}) [ppm]")

new_coord = res1.coord.with_(distance=np.linspace(0, lidar_coord.distance[-1], 1000))

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.grid(False)

ax.plot(
    new_coord.distance,
    utils.number_density_to_ppm(
        env.number_density_at(new_coord.x, 0, new_coord.z)["SO2"],
        new_coord.z
    ),
    label=r"$SO_2$ setting",
    c="black",
    ls="--",
    zorder=5
)
ax.plot(
    res1.coord.distance,
    n_cf_ignore,
    label="No correct.",
    c=cm.viridis(0/4)
)
ax.plot(
    res1.coord.distance,
    n_cf_ignore,
    label=f"Est at {0} ppm",
    c=cm.viridis(1/4)
)
ax.plot(
    res1.coord.distance,
    n_cf_1,
    label=f"Est at {est[0]} ppm",
    c=cm.viridis(2/4)
)
ax.plot(
    res1.coord.distance,
    n_cf_2,
    label=f"Est at {est[1]} ppm",
    c=cm.viridis(3/4)
)
ax.plot(
    res1.coord.distance,
    n_cf_3,
    label=f"Est at {est[2]} ppm",
    c=cm.viridis(4/4)
)
ax.set(
    xlabel="(Line of Sight) Distance [m]",
    ylabel=r"SO_2 Concentration [ppm]",
    xlim=(0, lidar_coord.distance[-1]),
    # ylim=(0, 40),
)
ax.legend(loc="upper left")
fig.savefig("samples/sim_result/sim_02_2_contami_result.pdf", dpi=400)
plt.show(block=False)
input("PRESS ANY KEY...")
