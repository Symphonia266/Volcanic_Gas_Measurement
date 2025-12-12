# coding: utf-8
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as np_SWV
from matplotlib import pyplot as plt
from pathlib import Path

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from gas_simulation import utils
from gas_simulation.consts import main_gases_props
from gas_simulation.atom import alphas_mol, alphas_aer
from gas_simulation.atom import betas_N2, betas_O2

from gas_simulation.model import (
    Field,
    Source,
    gen_fauntainsource,
    Gas,
    PlumeEnvironment,
)
from gas_simulation.lidar_model.lidar import Coord, Lidar, Dial, calc_dial_correction_factor
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

t_sec = 60*30 #[minutes]
field = Field(2, weather="clear", wind_direction_deg=75)
lidar = Lidar(M=100*t_sec)
lidar_coord = Coord(
    distance=np.arange(5, 1000, 5),
    theta_deg=0.0,
    x0=0,
    z0=1000,
)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -10], N_pt=30)
src = np.array([
    [1, 300, -50,  lidar_coord.z0 + 2], 
    [1, 500, -100, lidar_coord.z0 + 2], 
    [1, 700, -50,  lidar_coord.z0 + 2], 
    [1, 725, -100, lidar_coord.z0 + 2], 
    [1, 750, -50,  lidar_coord.z0 + 2], 
]).T
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
    time= t_sec,
    gas={
        # 煙源直下で30ppmになるよう調整したQ
        # "SO2": Gas(Q=80e5, offset=0, cross_section=xs_SO2),
        # "H2S": Gas(Q=40e5, offset=0, cross_section=xs_H2S),
        # 今回の煙源位置、風プロファイル設定で30ppm程度になるよう調整したQ
        "SO2": Gas(Q= 15e7, offset=0,     cross_section=xs_SO2),
        "H2S": Gas(Q=7.5e7, offset=0,     cross_section=xs_H2S),
        "O3" : Gas(Q=0,    offset=0.005, cross_section=xs_O3),
    },
)
env.show_gases(lidar_coord)

laser = np.arange(240, 370, 0.02)
wl = {
    "laser": laser,
    "N2_st": utils.wl_shift(laser, main_gases_props.at["N2", "sft"], False),
    "O2_st": utils.wl_shift(laser, main_gases_props.at["O2", "sft"], False),
    "N2_as": utils.wl_shift(laser, main_gases_props.at["N2", "sft"], True),
    "O2_as": utils.wl_shift(laser, main_gases_props.at["O2", "sft"], True),
}

# === Below this a provisional simulation scenario. ===
wl_laser = wl["laser"]
wl_N2 = wl["N2_as"]
wl_O2 = wl["O2_as"]
idx_320nm = np.searchsorted(wl_laser, 320)

beta_N2 = betas_N2(lidar_coord.z)
beta_O2 = betas_O2(lidar_coord.z)
tau_laser = env.transmittance(lidar_coord, wl_laser)
tau_N2 = tau_laser * env.transmittance(lidar_coord, wl_N2)
tau_O2 = tau_laser * env.transmittance(lidar_coord, wl_O2)

# === power calculation ===
p_N2 = lidar.power(
    dist=lidar_coord.distance[:, np.newaxis],
    wl=wl_laser[np.newaxis, :],
    beta_tau=beta_N2[:, np.newaxis] * tau_N2,
)
p_O2 = lidar.power(
    dist=lidar_coord.distance[:, np.newaxis],
    wl=wl_laser[np.newaxis, :],
    beta_tau=beta_O2[:, np.newaxis] * tau_O2,
)

xs_N2 = env.gas_inventory["SO2"].cross_section(wl_N2)
xs_O2 = env.gas_inventory["SO2"].cross_section(wl_O2)
mask = xs_N2 > xs_O2
wl_on = np.where(mask, wl_N2, wl_O2)
wl_off = np.where(mask, wl_O2, wl_N2)
xs_on = np.where(mask, xs_N2, xs_O2)
xs_off = np.where(mask, xs_O2, xs_N2)
d_xs_SO2 = xs_on - xs_off

mask = mask[np.newaxis, :]
p_on = np.where(mask, p_N2, p_O2)
p_off = np.where(mask, p_O2, p_N2)
tau_on = np.where(mask, tau_N2, tau_O2)
tau_off = np.where(mask, tau_O2, tau_N2)

# === DIAL calculation ===
dial = Dial()
dial_coord = Coord(
    distance=(lidar_coord.distance[1:]+lidar_coord.distance[:-1])/2,
    theta_deg=lidar_coord.theta_deg,
    x0=lidar_coord.x0,
    z0=lidar_coord.z0,
)
dial_dR = np.diff(lidar_coord.distance)
res = dial.calc(
    p_on_R1=p_on[:-1],
    p_on_R2=p_on[1:],
    p_off_R1=p_off[:-1],
    p_off_R2=p_off[1:],
    dR=dial_dR[:, np.newaxis],
    d_xs=d_xs_SO2[np.newaxis, :],
)
stat_err = dial.stat_error(
    p_on_R1=p_on[:-1],
    p_on_R2=p_on[1:],
    p_off_R1=p_off[:-1],
    p_off_R2=p_off[1:],
    dR=dial_dR[:, np.newaxis],
    d_xs=d_xs_SO2[np.newaxis, :],
)
dial_correction_factor = calc_dial_correction_factor(
    env=env, 
    alt=dial_coord.z[:, np.newaxis],
    wl_on = wl_on[np.newaxis, :], 
    wl_off = wl_off[np.newaxis, :],
    d_xs= d_xs_SO2[np.newaxis, :]
)

# === test plot ===
fig1, axes1 = viewer.lidar_equation_result_viewer(
    coord=lidar_coord,
    tau_on_dist=tau_on[:, idx_320nm],
    tau_off_dist=tau_off[:, idx_320nm],
    p_on_dist=p_on[:, idx_320nm],
    p_off_dist=p_off[:, idx_320nm],
    wl=wl_laser,
    tau_on_wl=tau_on[-1, :],
    tau_off_wl=tau_off[-1, :],
    p_on_wl=p_on[-1, :],
    p_off_wl=p_off[-1, :],
)
plt.show(block=False)

n_gas = env.number_density_at(lidar_coord.x, 0, lidar_coord.z)
_dial_coord = Coord(
    distance=(lidar_coord.distance[1:]+lidar_coord.distance[:-1])/2,
    theta_deg=lidar_coord.theta_deg,
    x0 = lidar_coord.x0, 
    z0 = lidar_coord.z0
)
n_true = {k:(v[1:]+v[:-1])/2 for k, v in n_gas.items()}

fig2, axes2 = viewer.dial_equation_result_viewer(
    env=env, 
    coord=dial_coord,
    n_true_dist=n_true["SO2"],
    res1_dist=res[:, idx_320nm], 
    res2_dist=(res-dial_correction_factor)[:, idx_320nm], 
    stat_err_dist=stat_err[:, idx_320nm], 
    wl=wl_laser, 
    n_true_wl=n_true["SO2"][-1],
    res1_wl=res[-1, :],
    res2_wl=(res-dial_correction_factor)[-1, :],
    stat_err_wl=stat_err[-1, :]
)

# 信号処理変更
window_N = 5
p_on_sw = np_SWV(p_on, window_shape=window_N, axis=0).sum(axis=-1)
p_off_sw = np_SWV(p_off, window_shape=window_N, axis=0).sum(axis=-1)
p_on_sw_R1 = p_on_sw[:-window_N]
p_on_sw_R2 = p_on_sw[window_N:]
p_off_sw_R1 = p_off_sw[:-window_N]
p_off_sw_R2 = p_off_sw[window_N:]

distance_sw = np_SWV(lidar_coord.distance, window_shape=window_N).mean(axis=-1)
dial_dR=distance_sw[window_N:] - distance_sw[:-window_N]
dial_coord = Coord(
    distance=(distance_sw[window_N:] + distance_sw[:-window_N]) / 2, 
    theta_deg=lidar_coord.theta_deg,
    x0 = lidar_coord.x0, 
    z0 = lidar_coord.z0
)

n_gas = env.number_density_at(lidar_coord.x, 0, lidar_coord.z)
n_true = {}
for k, v in n_gas.items():
    n_true[k] = np_SWV(v, window_shape=window_N).mean(axis=-1)
    n_true[k] = (n_true[k][window_N:]+n_true[k][:-window_N])/2

res = dial.calc(
    p_on_R1=p_on_sw_R1,
    p_on_R2=p_on_sw_R2,
    p_off_R1=p_off_sw_R1,
    p_off_R2=p_off_sw_R2,
    dR=dial_dR[:, np.newaxis],
    d_xs=d_xs_SO2[np.newaxis, :],
)
stat_err = dial.stat_error(
    p_on_R1=p_on_sw_R1,
    p_on_R2=p_on_sw_R2,
    p_off_R1=p_off_sw_R1,
    p_off_R2=p_off_sw_R2,
    dR=dial_dR[:, np.newaxis],
    d_xs=d_xs_SO2[np.newaxis, :],
)
dial_correction_factor = calc_dial_correction_factor(
    env=env, 
    alt=dial_coord.z[:, np.newaxis],
    wl_on = wl_on[np.newaxis, :], 
    wl_off = wl_off[np.newaxis, :],
    d_xs= d_xs_SO2[np.newaxis, :]
)

fig3, axes3 = viewer.dial_equation_result_viewer(
    env=env, 
    coord=dial_coord,
    n_true_dist=n_true["SO2"],
    res1_dist=res[:, idx_320nm], 
    res2_dist=(res-dial_correction_factor)[:, idx_320nm], 
    stat_err_dist=stat_err[:, idx_320nm], 
    wl=wl_laser, 
    n_true_wl=n_true["SO2"][-1],
    res1_wl=res[-1, :],
    res2_wl=(res-dial_correction_factor)[-1, :],
    stat_err_wl=stat_err[-1, :]
)

plt.show(block=False)
input("PRESS ANY KEY...")
