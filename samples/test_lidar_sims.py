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

from gas_simulation.consts import main_gases_props
from gas_simulation.atom import alphas_mol, alphas_aer
from gas_simulation import utils
from gas_simulation.atom import betas_N2, betas_O2

from gas_simulation.model import (
    Field,
    Source,
    gen_fauntainsource,
    Gas,
    PlumeEnvironment,
)
from gas_simulation.lidar_model.lidar import Coord, Lidar, Dial

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

field = Field(10, stab_class="D", wind_direction_deg=45)
lidar = Lidar()

end = 1000.0
elevation_deg=0.0
alt_offset=1000
dR = 5
lidar_coord = Coord(
    distance=np.arange(dR, end, dR),
    theta_deg=elevation_deg,
    x0=0,
    z0=alt_offset,
)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -10], N_pt=30)
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[300, -50], N_pt=20)
He = np.full_like(q, lidar_coord.z0 + 2)
source = Source(q, x_src, y_src, He)
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[400, -100], N_pt=20)
He = np.full_like(q, lidar_coord.z0 + 2)
source.add(q, x_src, y_src, He)
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[600, -200], N_pt=20)
He = np.full_like(q, lidar_coord.z0 + 2)
source.add(q, x_src, y_src, He)

env = PlumeEnvironment(
    field=field,
    source=source,
    time=10 * 60,
    gas={
        "SO2": Gas(Q=60e5, offset=0, cross_section=xs_SO2),
        "H2S": Gas(Q=30e5, offset=0, cross_section=xs_H2S),
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
# (
#     to be expanded to support arbitrary wavelength inputs in the future;
#     script to be modified for the simulation API
# )
wl_laser = wl["laser"]
wl_N2 = wl["N2_as"]
wl_O2 = wl["O2_as"]
idx_300nm = np.searchsorted(wl_laser, 300)

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
    theta_deg=elevation_deg,
    x0=0,
    z0=alt_offset,
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
for n in utils.number_density_to_ppm(stat_err, dial_coord.z[:, np.newaxis])[:, idx_300nm]:
    print(n)

# === Preparations for DIAL Calculations  ===
alpha_mol_laser = alphas_mol(
    wl_laser[np.newaxis, :], 
    dial_coord.z[:, np.newaxis]
)
alpha_mol_on = alpha_mol_laser + alphas_mol(
    wl_on[np.newaxis, :], 
    dial_coord.z[:, np.newaxis]
)
alpha_mol_off = alpha_mol_laser + alphas_mol(
    wl_off[np.newaxis, :], 
    dial_coord.z[:, np.newaxis]
)
d_alpha_mol = alpha_mol_on - alpha_mol_off

alpha_aer_laser = alphas_aer(
    wl_laser[np.newaxis, :], 
    dial_coord.z[:, np.newaxis],
    env.aer_absorp_feat
)
alpha_aer_on = alpha_aer_laser + alphas_aer(
    wl_on[np.newaxis, :], 
    dial_coord.z[:, np.newaxis], 
    env.aer_absorp_feat
)
alpha_aer_off = alpha_aer_laser + alphas_aer(
    wl_off[np.newaxis, :], 
    dial_coord.z[:, np.newaxis], 
    env.aer_absorp_feat
)
d_alpha_aer = alpha_aer_on - alpha_aer_off

# n_gas = env.number_density_at(dial.x_grid, 0, dial.z_grid)
# alpha_H2S_laser =n_gas["H2S"] *env.gas_profile["H2S"].cross_section(wl_laser)[np.newaxis, :]
# alpha_H2S_on = (
#     alpha_H2S_laser +
#     n_gas["H2S"] *env.gas_profile["H2S"].cross_section(wl_on)[np.newaxis, :]
# )
# alpha_H2S_off = (
#     alpha_H2S_laser +
#     n_gas["H2S"] *env.gas_profile["H2S"].cross_section(wl_off)[np.newaxis, :]
# )
# d_alpha_H2S = alpha_H2S_on - alpha_H2S_off

dial_correction_factor = (d_alpha_mol + d_alpha_aer) / d_xs_SO2[np.newaxis, :]

# === test plot ===
fig, axes = plt.subplots(2, 2, layout="constrained")
axes[0, 0].plot(lidar_coord.distance, tau_on[:, idx_300nm], c="red", ls="-", label="on")
axes[0, 0].plot(
    lidar_coord.distance, tau_off[:, idx_300nm], c="blue", ls="-", label="off"
)
# axes[0].plot(env.lidar.distance[1:], tau_N2[:, idx_300nm],  c ="orange",ls="--", label="N2")
# axes[0].plot(env.lidar.distance[1:], tau_O2[:, idx_300nm],  c ="green", ls="--", label="O2")

axes[0, 1].plot(
    lidar_coord.distance, p_on[:, idx_300nm], marker="o", c="red", ls="-", label="on"
)
axes[0, 1].plot(
    lidar_coord.distance,
    p_off[:, idx_300nm],
    marker="o",
    c="blue",
    ls="-",
    label="off",
)

axes[1, 0].plot(wl_laser, tau_on[-1, :], c="red", ls="-", label="on")
axes[1, 0].plot(wl_laser, tau_off[-1, :], c="blue", ls="-", label="off")
# axes[0].plot(wl_laser, tau_N2[9, :],  c ="orange",ls="--", label="N2")
# axes[0].plot(wl_laser, tau_O2[9, :],  c ="green", ls="--", label="O2")

axes[1, 1].plot(wl_laser, p_on[-1, :], c="red", ls="-", label="on")
axes[1, 1].plot(wl_laser, p_off[-1, :], c="blue", ls="-", label="off")
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

plt.show(block=False)

fig, ax = plt.subplots(1, 1, layout="constrained")
ax.grid(which="major", ls="-", c="darkgrey")
ax.grid(which="minor", ls="--", c="lightgrey")

r = np.linspace(lidar_coord.distance.min(), lidar_coord.distance.max(), 1000)
x = np.linspace(lidar_coord.x.min(), lidar_coord.x.max(), 1000)
z = np.linspace(lidar_coord.z.min(), lidar_coord.z.max(), 1000)
ax.plot(
    r,
    utils.number_density_to_ppm(env.number_density_at(x, 0, z)["SO2"], z),
    c="black",
    label="True",
)
ax.scatter(
    dial_coord.distance,
    utils.number_density_to_ppm(
        env.number_density_at(dial_coord.x, 0, dial_coord.z)["SO2"], dial_coord.z
    ),
    c="black",
)
ax.errorbar(
    x=dial_coord.distance,
    y=utils.number_density_to_ppm(res, dial_coord.z[:, np.newaxis])[:, idx_300nm],
    yerr=utils.number_density_to_ppm(
        stat_err, dial_coord.z[:, np.newaxis]
    )[:, idx_300nm],
    capsize=8, fmt='o', markersize=8, ecolor='red', color='red',
    label="DIAL Result",
)
ax.errorbar(
    x=dial_coord.distance,
    y=utils.number_density_to_ppm(
        res - dial_correction_factor, dial_coord.z[:, np.newaxis]
    )[:, idx_300nm],
    yerr=utils.number_density_to_ppm(
        stat_err, dial_coord.z[:, np.newaxis]
    )[:, idx_300nm],
    capsize=8, fmt='o', markersize=8, ecolor='green', color='green',
    label="DIAL Result (mol. and aer. Correction)",
)
ax.set_xlabel("lidar distance [m]")
ax.set_ylabel(r"concentration $n_{SO2}$[ppm]")
ax.legend()
plt.show(block=False)

# 信号処理変更
window_N = 5
p_on_sw = np_SWV(p_on, window_shape=window_N, axis=0).sum(axis=-1)
p_off_sw = np_SWV(p_off, window_shape=window_N, axis=0).sum(axis=-1)
p_on_sw_R1 = p_on_sw[:-window_N]
p_on_sw_R2 = p_on_sw[window_N:]
p_off_sw_R1 = p_off_sw[:-window_N]
p_off_sw_R2 = p_off_sw[window_N:]

distance_sw = np_SWV(lidar_coord.distance, window_shape=window_N).mean(axis=-1)
dial_coord = Coord(
    distance=(distance_sw[window_N:] + distance_sw[:-window_N]) / 2, 
    theta_deg=elevation_deg,
    x0 = 0, 
    z0 = alt_offset
)
dial_dR=distance_sw[window_N:] - distance_sw[:-window_N]

n_gas = env.number_density_at(lidar_coord.x, 0, lidar_coord.z)
n_true_sw = np_SWV(n_gas["SO2"], window_shape=window_N).mean(axis=-1)

print(f"{dial_coord.distance.shape} : {dial_coord.distance}")
print(f"{dial_dR.shape} : {dial_dR}")
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
print(res.shape)
print(dial_dR.shape)
# === Preparations for DIAL Calculations  ===
alpha_mol_laser = alphas_mol(wl_laser[np.newaxis, :], dial_coord.z[:, np.newaxis])
alpha_mol_on = alpha_mol_laser + alphas_mol(
    wl_on[np.newaxis, :], dial_coord.z[:, np.newaxis]
)
alpha_mol_off = alpha_mol_laser + alphas_mol(
    wl_off[np.newaxis, :], dial_coord.z[:, np.newaxis]
)
d_alpha_mol = alpha_mol_on - alpha_mol_off

alpha_aer_laser = alphas_aer(
    wl_laser[np.newaxis, :], dial_coord.z[:, np.newaxis], env.aer_absorp_feat
)
alpha_aer_on = alpha_aer_laser + alphas_aer(
    wl_on[np.newaxis, :], dial_coord.z[:, np.newaxis], env.aer_absorp_feat
)
alpha_aer_off = alpha_aer_laser + alphas_aer(
    wl_off[np.newaxis, :], dial_coord.z[:, np.newaxis], env.aer_absorp_feat
)
d_alpha_aer = alpha_aer_on - alpha_aer_off

# n_gas = env.number_density_at(dial.x_grid, 0, dial.z_grid)
# alpha_H2S_laser =n_gas["H2S"] *env.gas_profile["H2S"].cross_section(wl_laser)[np.newaxis, :]
# alpha_H2S_on = (
#     alpha_H2S_laser +
#     n_gas["H2S"] *env.gas_profile["H2S"].cross_section(wl_on)[np.newaxis, :]
# )
# alpha_H2S_off = (
#     alpha_H2S_laser +
#     n_gas["H2S"] *env.gas_profile["H2S"].cross_section(wl_off)[np.newaxis, :]
# )
# d_alpha_H2S = alpha_H2S_on - alpha_H2S_off

dial_correction_factor = (d_alpha_mol + d_alpha_aer) / d_xs_SO2[np.newaxis, :]

fig, axes = plt.subplots(1, 2, layout="constrained")
axes[0].plot(
    lidar_coord.distance, 
    p_on[:, idx_300nm], 
    marker="o", c="red", ls="--", label="on"
)
axes[0].plot(
    lidar_coord.distance,
    p_off[:, idx_300nm],
    marker="o",
    c="blue",
    ls="--",
    label="off",
)
axes[0].plot(
    distance_sw,
    p_on_sw[:, idx_300nm],
    marker="*",
    c="red",
    ls="-",
    label="on (5pt sum)",
)
axes[0].plot(
    distance_sw,
    p_off_sw[:, idx_300nm],
    marker="*",
    c="blue",
    ls="-",
    label="off (5pt sum)",
)
# ==========================================

axes[1].plot(
    r,
    utils.number_density_to_ppm(env.number_density_at(x, 0, z)["SO2"], z),
    c="black",
    label="True",
)
# a = Coord(distance_sw, lidar.coord.theta_deg, lidar.coord.x0, lidar.coord.z0)
# axes[1].scatter(
#     lidar.coord.distance,
#     utils.number_density_to_ppm(n_gas["SO2"], lidar.coord.z),
#     c="black",
#     marker="o",
#     s=10,
# )

# axes[1].scatter(
#     a.distance,
#     utils.number_density_to_ppm(n_true_sw, a.z),
#     c="black",
#     marker="o",
# )

n_true_sw = (n_true_sw[window_N:] + n_true_sw[:-window_N]) / 2
axes[1].scatter(
    dial_coord.distance,
    utils.number_density_to_ppm(n_true_sw, dial_coord.z),
    c="black",
    marker="*",
)
axes[1].errorbar(
    x=dial_coord.distance,
    y=utils.number_density_to_ppm(res, dial_coord.z[:, np.newaxis])[:, idx_300nm],
    yerr=utils.number_density_to_ppm(
        stat_err, dial_coord.z[:, np.newaxis]
    )[:, idx_300nm],
    capsize=8, fmt='o', markersize=8, ecolor='red', color='red',
    label="DIAL Result",
)
axes[1].errorbar(
    x=dial_coord.distance,
    y=utils.number_density_to_ppm(
        res - dial_correction_factor, dial_coord.z[:, np.newaxis]
    )[:, idx_300nm],
    yerr=utils.number_density_to_ppm(
        stat_err, dial_coord.z[:, np.newaxis]
    )[:, idx_300nm],
    capsize=8, fmt='o', markersize=8, ecolor='green', color='green',
    label="DIAL Result (mol. and aer. Correction)",
)

axes[0].set_ylabel(r"receive power $P_{phot}$")
axes[1].set_ylabel(r"concentration $n_{SO2}$[ppm]")
axes[0].set_yscale("log")

for ax in axes:
    ax.grid(which="major", ls="-", c="darkgrey")
    ax.grid(which="minor", ls="--", c="lightgrey")
    ax.set_xlabel("lidar distance [m]")
    ax.legend()

plt.show(block=False)
input("PRESS ANY KEY...")
