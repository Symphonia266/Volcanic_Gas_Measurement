# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from diffusion_model.diffuse_plume import DiffusePlumeLidar

# ---- パラメータ ----
model_name = "pasquill"
windspeed = 2
wether = "clear"
stab_class = None

wind_direction_deg = 90   # +xから反時計回り（例：30° → 風が右から左に30°傾く）
elevation_deg = 0   # +xから反時計回り（例：30° → 風が右から左に30°傾く）

# ---- 計算グリッド（※全てライダー座標系で定義！）----
x_grid = np.array([0.0])
y_grid = np.array([0.0])
z_grid = np.array([0.0])

model = DiffusePlumeLidar(model_name, windspeed, wind_direction_deg, elevation_deg, wether, stab_class)

# ---- ソース設定（ライダー座標系で記述）----
model.entry_source(1.0, 10.0, 0.0, 0.0)

print(model.core.sources)
input()

# 他の点源（例）
# model.entry_source(1, 20, 20, 0)
# model.entry_source(1, 5, 5, 0)
# model.entry_source(1, 10, -10, 0)

# ---- 濃度計算 ----
C = model.Concentration(x_grid, y_grid, z_grid)
C_grid = model.Concentration(x_grid, 0, 0)

dR=30
x_obs = np.arange(x_grid.min(), x_grid.max(), dR)
C_obs = model.Concentration(x_obs, 0, 0)

# ---- プロット ----
fig, ax = plt.subplots(1, 2, constrained_layout=True)
ax[1].plot(x_grid, C_grid)
ax[1].scatter(x_obs, C_obs)
# ax[0].set_title(f"Ground Concentration Map (z=0)\nWind Direction = {wind_direction_deg}°")
ax[1].set_xlabel("horizontal distance[m]")
ax[1].set_ylabel("concentration coefficient")
ax[1].set_ylim(0, None)

ymin, ymax = ax[1].get_ylim()
im = ax[0].imshow(
    C,
    origin='lower',
    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
    aspect='equal',
    vmax=ymax,
    vmin=ymin,
    cmap='jet'
)
fig.colorbar(im, label='Concentration')

ax[0].set_xlabel("Lidar X-axis [m]")
ax[0].set_ylabel("Lidar Y-axis [m]")
plt.show(block=False)
input()
