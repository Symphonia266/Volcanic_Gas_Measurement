# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusion_model.diffuse_plume import DiffusePlume
from diffusion_model.diffuse_plume import DiffusePlumeLidar

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

windspeed = 2
wether = "clear"
model = DiffusePlume("pasquill", windspeed, wether, None)

x_g = np.linspace(0, 100, 250) + 1e-6
y_g = np.linspace(-50, 50, 250)
z_g = np.array([0])

# 面源再現
Q = 1
r = 5
pos_cnt = [10, 0]
X, Y, Z = np.meshgrid(x_g, y_g, z_g)
mask = ((X-pos_cnt[0])**2 + (Y-pos_cnt[1])**2) <= r**2
x_src = X[mask]
y_src = Y[mask]
z_src = Z[mask]
q_src = 10/mask.sum()

model.entry_source(q_src, x_src, y_src, z_src)

# # 他点源
# model.entry_source(Q, 20, 20, 0)
# model.entry_source(Q, 5, 5, 0)
# model.entry_source(Q, 10, -10, 0)

C = model.Concentration(X, Y, Z)

fig, ax = plt.subplots(1,1)
im = ax.imshow(
    C,          # 軸を転置して y を縦軸に
    # norm=LogNorm(),
    origin='lower',      # 左下を (x_min, y_min) に
    extent=[x_g.min(), x_g.max(), y_g.min(), y_g.max()],
    aspect='equal',
    vmax= 0.02,
    cmap='jet'
)
fig.colorbar(im, label='Concentration [units]')
ax.set_xlabel("downwind distance [m]")
ax.set_ylabel("horizontal spread [m]")
ax.set_title("Ground-level Concentration Heatmap (z=0)")
plt.show(block=False)
input()
