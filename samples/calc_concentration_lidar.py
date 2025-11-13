# coding: utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gas_simulation.diffusion_model.diffuse_plume import DiffusePlumeLidar


def plotter(x, y, C, func, x_obs):
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    im = ax[0].imshow(
        C,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        vmax=0.025,
        vmin=0,
        cmap="jet",
    )
    fig.colorbar(im, label="Concentration")
    ax[0].set_xlabel("Lidar X-axis [m]")
    ax[0].set_ylabel("Lidar Y-axis [m]")
    ax[1].plot(x, func(x))
    ax[1].scatter(x_obs, func(x_obs))
    plt.show(block=False)

x_grid = np.linspace(0, 100, 250)
y_grid = np.linspace(-50, 50, 250)
z_grid = np.array([0])  # 地表面
dR = 7.5
x_obs = np.arange(x_grid.min(), x_grid.max(), dR)[1:]

# ---- パラメータ ----
model_name = "pasquill"
windspeed = 2
wether = "clear"
stab_class = None

# 半径5mの面源
r = 5
x_src = np.linspace(-r, r, 100)
y_src = np.linspace(-r, r, 100)
x_src, y_src = np.meshgrid(x_src, y_src, indexing="ij")
z_src = 0

mask = (x_src ** 2 + y_src**2) <= r**2
x_src = x_src[mask].flatten()
y_src = y_src[mask].flatten()
q_src = 1 / mask.sum()
H_src = 2

# ---- ソース設定（ライダー座標系で記述）----
model = DiffusePlumeLidar(model_name, windspeed, 45, 0, wether=wether, stab_class=stab_class)
pos_cnt = [50, -20, 0]
model.entry_source(q_src, x_src + pos_cnt[0], y_src + pos_cnt[1], z_src + pos_cnt[2], H_src)
C = model.Concentration(x_grid, y=y_grid, z=z_grid)
plotter(x_grid, y_grid, C, model.Concentration, x_obs)
model.clear_source()

model.update_azim_elev(90, 0)
pos_cnt = [50, -50, 0]
model.entry_source(q_src, x_src + pos_cnt[0], y_src + pos_cnt[1], z_src + pos_cnt[2], H_src)
C = model.Concentration(x_grid, y=y_grid, z=z_grid)
plotter(x_grid, y_grid, C, model.Concentration, x_obs)
model.clear_source()

# fig = plt.figure(constrained_layout=True)
# ax = fig.add_subplot(111, projection="3d")
# ax.view_init(elev=15, azim=20)
# ax.plot_wireframe(
#     X[:, :, 0], Y[:, :, 0], C[:, :, 0], color="blue", rstride=5, cstride=5
# )
# ax.set_xlabel("X axis")
# ax.set_ylabel("Y axis")
# ax.set_zlabel("concentration coefficient")
plt.show(block=False)
input("ENTER ANY KEY")
