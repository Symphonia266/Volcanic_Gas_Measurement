# coding: utf-8
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from gas_simulation.consts import main_gases_props
from gas_simulation.atom import alphas_mol, alphas_aer
from gas_simulation import utils
from gas_simulation.diffusion_model.func import gen_fauntainsource
from gas_simulation.atom import betas_N2, betas_O2

from gas_simulation.model import Field, Source, Gas, Environment
from gas_simulation.lidar_model.lidar import Lidar, Dial

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

field = Field(2, weather="clear", wind_direction_deg=0)
lidar = Lidar(end=100.0, alt_offset=1000)
# q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -10], N_pt=30)
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[-10, 0], N_pt=30)
He = np.full_like(q, lidar.alt_offset + 2)
source = Source(q, x_src, y_src, He)
time = 10 * 60  # 10 minutes
gas = {
    "SO2": Gas(Q=30e5, offset=0, cross_section=xs_SO2),
    "H2S": Gas(Q=15e5, offset=0, cross_section=xs_H2S),
}

env = Environment(
    field=field,
    source=source,
    lidar=lidar,
    time=time,
    gas=gas
)
env.show_gases()

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

beta_N2 = betas_N2(env.lidar.z_grid[1:])
beta_O2 = betas_O2(env.lidar.z_grid[1:])
tau_laser = env.transmittance(wl_laser)
tau_N2 = tau_laser*env.transmittance(wl_N2)
tau_O2 = tau_laser*env.transmittance(wl_O2)

# === power calculation ===
p_N2 = env.lidar.power(
        wl=wl_laser,
        beta_tau=beta_N2[:, np.newaxis] * tau_N2,
)
p_O2 = env.lidar.power(
        wl=wl_laser,
        beta_tau=beta_O2[:, np.newaxis] * tau_O2,
)
xs_N2 =env.gas_profile["SO2"].cross_section(wl_N2)
xs_O2 =env.gas_profile["SO2"].cross_section(wl_O2)
mask = xs_N2 > xs_O2
wl_on = np.where(mask, wl_N2, wl_O2)
wl_off = np.where(mask, wl_O2, wl_N2)
xs_on = np.where(mask, xs_N2, xs_O2)
xs_off = np.where(mask, xs_O2, xs_N2)
d_xs_SO2 = xs_on - xs_off

mask = mask[np.newaxis, :]
p_on    = np.where(mask, p_N2, p_O2)
p_off   = np.where(mask, p_O2, p_N2)
tau_on  = np.where(mask, tau_N2, tau_O2)
tau_off = np.where(mask, tau_O2, tau_N2)

# === DIAL calculation ===
dial = Dial(env.lidar)
print(dial.distance)
print(dial.dR)
res = dial.concentration(p_on, p_off, dial.dR[:, np.newaxis], d_xs_SO2[np.newaxis, :])

# === test plot ===
idx_300nm = np.searchsorted(wl_laser, 300)
print(utils.number_density_to_ppm(res[:, idx_300nm], dial.z_grid))

fig, axes = plt.subplots(1,2, layout="constrained")
axes[0].plot(env.lidar.distance[1:], tau_on[:, idx_300nm],  c ="red",   ls="-",  label="on")
axes[0].plot(env.lidar.distance[1:], tau_off[:, idx_300nm], c ="blue",  ls="-",  label="off")
# axes[0].plot(env.lidar.distance[1:], tau_N2[:, idx_300nm],  c ="orange",ls="--", label="N2")
# axes[0].plot(env.lidar.distance[1:], tau_O2[:, idx_300nm],  c ="green", ls="--", label="O2")
axes[0].set_ylabel(r"transmittance $\tau$")
axes[0].set_ylim(0, 1.0)

axes[1].plot(env.lidar.distance[1:], p_on[:, idx_300nm],  c ="red",   ls="-",  label="on")
axes[1].plot(env.lidar.distance[1:], p_off[:, idx_300nm], c ="blue",  ls="-",  label="off")
# axes[1].plot(env.lidar.distance[1:], p_N2[:, idx_300nm],  c ="orange",ls="--", label="N2")
# axes[1].plot(env.lidar.distance[1:], p_O2[:, idx_300nm],  c ="green", ls="--", label="O2")
axes[1].set_ylabel(r"$power_{phot}$")

# axes[0].set_yscale("log")
axes[1].set_yscale("log")
for ax in axes:
    ax.grid(which="major", ls="-", c="darkgrey")
    ax.grid(which="minor", ls="--", c="lightgrey")
    ax.set_xlabel("lidar distance [m]")
    ax.legend()

plt.show(block=False)

fig, axes = plt.subplots(1,2, layout="constrained")
axes[0].plot(wl_laser, tau_on[9,  :],  c ="red",   ls="-",  label="on")
axes[0].plot(wl_laser, tau_off[9, :], c ="blue",  ls="-",  label="off")
# axes[0].plot(wl_laser, tau_N2[9, :],  c ="orange",ls="--", label="N2")
# axes[0].plot(wl_laser, tau_O2[9, :],  c ="green", ls="--", label="O2")
axes[0].set_ylabel(r"transmittance $\tau$")
axes[0].set_ylim(0, 1.0)

axes[1].plot(wl_laser, p_on[9,  :],  c ="red",   ls="-",  label="on")
axes[1].plot(wl_laser, p_off[9, :], c ="blue",  ls="-",  label="off")
# axes[1].plot(wl_laser, p_N2[9, :],  c ="orange",ls="--", label="N2")
# axes[1].plot(wl_laser, p_O2[9, :],  c ="green", ls="--", label="O2")
axes[1].set_ylabel(r"received power $P_{phot}$")

# axes[0].set_yscale("log")
axes[1].set_yscale("log")
for ax in axes:
    ax.grid(which="major", ls="-", c="darkgrey")
    ax.grid(which="minor", ls="--", c="lightgrey")
    ax.set_xlabel("laser wavelength [nm]")
    ax.legend()

plt.show(block=False)

# a = utils.number_density_to_ppm(res, dial.z_grid[:, np.newaxis])
# # b = utils.number_density_to_ppm(res-dial_correction_factor, dial.z_grid)

fig, ax = plt.subplots(1,1)
ax.grid(which="major", ls="-", c="darkgrey")
ax.grid(which="minor", ls="--", c="lightgrey")

r = np.linspace(lidar.distance.min(), lidar.distance.max(),1000)
x = np.linspace(lidar.x_grid.min(), lidar.x_grid.max(),1000)
z = np.linspace(lidar.z_grid.min(), lidar.z_grid.max(),1000)
ax.plot(
    r, 
    utils.number_density_to_ppm(env.distribution(x, 0, z)["SO2"], z),
    c="black", label="True"
)
ax.scatter(
    dial.distance, 
    utils.number_density_to_ppm(env.distribution(dial.x_grid, 0, dial.z_grid)["SO2"], dial.z_grid),
    c="black"
)
ax.scatter(
    dial.distance, 
    utils.number_density_to_ppm(res, dial.z_grid[:, np.newaxis])[:, idx_300nm],
    c="red", label="DIAL Result"
)
# ax.scatter(dial.distance, b[:, idx_300nm], c="blue", label="corrected DIAL Result")

ax.set_xlabel("lidar distance [m]")
ax.set_ylabel("concentration [ppm]")
ax.legend()

plt.show(block=False)

# # === Preparations for DIAL Calculations  ===
# alpha_mol_laser = alphas_mol(wl_laser[np.newaxis, :], dial.z_grid[:, np.newaxis])
# alpha_mol_on  = (
#     alpha_mol_laser + 
#     alphas_mol(wl_on[np.newaxis, :], dial.z_grid[:, np.newaxis]) 
# )
# alpha_mol_off = (
#     alpha_mol_laser + 
#     alphas_mol(wl_off[np.newaxis, :], dial.z_grid[:, np.newaxis])
# )
# d_alpha_mol = alpha_mol_on - alpha_mol_off

# alpha_aer_laser = alphas_aer(
#     wl_laser[np.newaxis, :],
#     dial.z_grid[:, np.newaxis], 
#     env.aer_absorp_feat
# )
# alpha_aer_on    = (
#     alpha_aer_laser + 
#     alphas_aer(
#         wl_on[np.newaxis, :], 
#         dial.z_grid[:, np.newaxis], 
#         env.aer_absorp_feat
#     )
# )
# alpha_aer_off   = (
#     alpha_aer_laser + 
#     alphas_aer(
#         wl_off[np.newaxis, :], 
#         dial.z_grid[:, np.newaxis], 
#         env.aer_absorp_feat
#     )
# )
# d_alpha_aer = alpha_aer_on - alpha_aer_off

# C = env.plume_model.concentration(dial.x_grid, 0, dial.z_grid, time=env.time)
# n_H2S = env.distribution(C, env.gas_profile["H2S"])[:, np.newaxis] 
# alpha_H2S_laser =n_H2S *env.gas_profile["H2S"].cross_section(wl_laser)[np.newaxis, :]
# alpha_H2S_on = (
#     alpha_H2S_laser + 
#     n_H2S *env.gas_profile["H2S"].cross_section(wl_on)[np.newaxis, :]
# )
# alpha_H2S_off = (
#     alpha_H2S_laser +
#     n_H2S *env.gas_profile["H2S"].cross_section(wl_off)[np.newaxis, :]
# )
# d_alpha_H2S = alpha_H2S_on - alpha_H2S_off

# dial_correction_factor = (d_alpha_mol + d_alpha_aer + d_alpha_H2S)/d_xs_SO2

input("PRESS ANY KEY...")
