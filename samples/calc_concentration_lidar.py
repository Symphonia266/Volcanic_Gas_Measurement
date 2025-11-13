# coding: utf-8
import os
from re import X
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gas_simulation.diffusion_model.diffuse_plume import DiffusePlumeLidar
from gas_simulation.diffusion_model.func import gen_fauntainsource


def plotter(x, y, C, func, x_obs):
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    im = ax[0].imshow(
        C,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        vmax=0.01,
        vmin=0,
        cmap="jet",
    )
    fig.colorbar(im, label="Concentration")
    ax[0].set_xlabel("Lidar X-axis [m]")
    ax[0].set_ylabel("Lidar Y-axis [m]")
    ax[1].plot(x, func(x))
    ax[1].scatter(x_obs, func(x_obs))
    ax[1].set_xlabel("distance [m]")
    ax[1].set_ylabel("Concentration coefficient")
    ax[1].set_ylim(0, 0.01)

    fig2 = plt.figure(constrained_layout=True)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.view_init(elev=15, azim=20)

    X, Y = np.meshgrid(x_grid, y_grid)
    ax2.plot_wireframe(
        X, Y, C, color="blue", rstride=5, cstride=5
    )
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.set_zlabel("concentration coefficient")
    ax2.set_zlim(0, 0.01)
    plt.show(block=False)

x_grid = np.linspace(0, 100, 200)
y_grid = np.linspace(-50, 50, 200)
x_grid = x_grid[:, np.newaxis]
y_grid = y_grid[np.newaxis, :]
z_grid = 0  # 地表面
dR = 7.5
x_obs = np.arange(x_grid.min(), x_grid.max(), dR)[1:]

# plt.figure()
# x_src, y_src, z_src, q_src = gen_fauntainsource(5, (50, -40, 0), 100)
# plt.scatter(x_src, y_src, c=q_src)
# plt.show(block=False)
# input("ENTER ANY KEY...")
# plt.cla()
# x_src, y_src, z_src, q_src = gen_fauntainsource(5, (50, -40, 0), 10)
# plt.scatter(x_src, y_src, c=q_src)
# plt.show(block=False)
# input("ENTER ANY KEY...")

H_src = 2

# ---- ソース設定（ライダー座標系で記述）----
model = DiffusePlumeLidar("pasquill", 2, 90, stab_class="A")
x_src, y_src, z_src, q_src = gen_fauntainsource(5, (50, -40, 0), 50)
model.entry_source(q_src, x_src, y_src, z_src, H_src)
C = model.Concentration(x_grid, y=y_grid, z=z_grid)
plotter(x_grid, y_grid, C, model.Concentration, x_obs)
model.clear_source()

x_src, y_src, z_src, q_src = gen_fauntainsource(5, (50, -40, 0), 5)
model.entry_source(q_src, x_src, y_src, z_src, H_src)
C = model.Concentration(x_grid, y=y_grid, z=z_grid)
plotter(x_grid, y_grid, C, model.Concentration, x_obs)
model.clear_source()

model = DiffusePlumeLidar("pasquill", 10, 90, stab_class="D")
x_src, y_src, z_src, q_src = gen_fauntainsource(5, (50, -40, 0), 50)
model.entry_source(q_src, x_src, y_src, z_src, H_src)
C = model.Concentration(x_grid, y=y_grid, z=z_grid)
plotter(x_grid, y_grid, C, model.Concentration, x_obs)
model.clear_source()

# model.update_azim_elev(90, 0)
# pos_cnt = [0, 0, 0]
# model.entry_source(q_src, x_src + pos_cnt[0], y_src + pos_cnt[1], z_src + pos_cnt[2], H_src)
# C = model.Concentration(x_grid, y=y_grid, z=z_grid)
# plotter(x_grid, y_grid, C, model.Concentration, x_obs)
# model.clear_source()

# plt.show(block=False)
input("ENTER ANY KEY")
