import sys
from matplotlib.pylab import imshow
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from gas_simulation.diffusion_model.func import gen_fauntainsource
from gas_simulation.diffusion_model.pasquill_stable_classfication import (
    classification_table,
    classification_label,
    classify_atomosphere_stability,
)
from gas_simulation.diffusion_model.pasquill_model.pasquill_gifford_spreadwidth import connect_smoothly_multi, weight
from gas_simulation.diffusion_model.pasquill_model.pasquill_gifford_spreadwidth import SpreadWidth as PasquillSpread
from gas_simulation.diffusion_model.diffuse_plume import (
    Field, 
    Source, 
    DiffusePlume, 
    DiffusePlumeLidar
)
# from diffusion_model.sutton_model.sutton_spreadwidth import SpreadWidth as SuttonSpread


# =======================================================================================

windspeed = 5
wether = "clear"
# print(classification_table)
print(classification_label)
stab_class = classify_atomosphere_stability(windspeed, wether)
print(f"Atomospheric stability is : {stab_class}")

"""
これは現在の風速, 日照（＝天気）から, パスキルの大気安定度分類に基づいて安定分類を行う関数です. 
パスキルの拡散幅モデル, サットンの拡散幅モデルにこの分類が関連します. 
"""

# ========================================================================================

spread_model = PasquillSpread()
i = np.linspace(0, 7)
x = np.power(10, i)
fig, axes = plt.subplots(1, 2)
for ax in axes.ravel():
    ax.grid(which="major", ls="-", c="darkgrey")
    ax.grid(which="minor", ls="--", c="lightgrey")
    ax.set_xlabel("downwind distance [m]")
    ax.set_xscale("log")
    ax.set_yscale("log")

spread_y = {}
spread_z = {}

for lb in classification_label:
    spread_y[lb] = spread_model.lateral(x, lb)
    spread_z[lb] = spread_model.vertical(x, lb)
    axes[0].plot(x, spread_y[lb], ls="--", label=lb)
    axes[1].plot(x, spread_z[lb], ls="--", label=lb)

axes[0].set_ylabel("lateral spread width [m / 3 min]")
axes[1].set_ylabel("vertical spread width [m / 3 min]")
axes[0].legend()
plt.show(block=False)

"""
これは点源から湧出する気体が一定の風速・風向によって形成する煙流の鉛直・水平方向の拡散幅を導出するクラスです. 
所属するメソッドとして, 水平方向の拡散幅を返すlateral, 鉛直方向の拡散幅を返すverticalが存在します.
これは安定分類クラスを継承しており, 安定度が高いほど拡散幅は水平方向・鉛直方向共に減少する傾向があります. 
拡散幅はPasquill-Gifford線図を手動でサンプリングしたプロットデータから
近傍, 遠方のデータを切り取ってlog-log線形回帰による外挿を施すことで
100mから100,000m区間のデータしかなかった原著から拡張しています. 
また, 外挿関数部とサンプリングデータの線形補完部の接続は対数重みによるスムージングを行っています. 
"""

# =======================================================================

field = Field(windspeed=2, stab_class="A")

q, x_src, y_src = gen_fauntainsource(radius=3, cnt=[50, 30], N_pt=5)
He = np.full_like(q, 2)
source = Source(q, x_src, y_src, He)

model = DiffusePlume(field, source)
x = np.linspace(0, 100, 100)[np.newaxis, :]
y = np.linspace(-50, 50, 100)[:, np.newaxis]
z = np.array([0])
C = model.Concentration(x, y, z)

fig2, ax2 = plt.subplots()
im = ax2.imshow(
    C,          # 軸を転置して y を縦軸に
    # norm=LogNorm(),
    origin='lower',      # 左下を (x_min, y_min) に
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='equal',
    cmap='jet'
)
fig2.colorbar(im, label='Concentration [units]')
ax2.set_xlabel("downwind distance [m]")
ax2.set_ylabel("horizontal spread [m]")
ax2.set_title("Ground-level Concentration Heatmap (z=0)")
plt.show(block=False)

# =====================================================================

field = Field(2, weather="clear")
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -30], N_pt=10)
He = np.full_like(q, 2)
source = Source(q, x_src, y_src, He)
model = DiffusePlumeLidar(field, 90, source)

x = np.linspace(0, 100, 250)[np.newaxis, :]
y = np.linspace(-50, 50, 250)[:, np.newaxis]
z = np.array([0])
C = model.Concentration(x, y, z)

fig, ax = plt.subplots(1,1)
im = ax.imshow(
    C,          # 軸を転置して y を縦軸に
    # norm=LogNorm(),
    origin='lower',      # 左下を (x_min, y_min) に
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='equal',
    cmap='jet'
)
fig.colorbar(im, label='Concentration [units]')
ax.set_xlabel("downwind distance [m]")
ax.set_ylabel("horizontal spread [m]")
ax.set_title("Ground-level Concentration Heatmap (z=0)")
plt.show(block=False)

# ====================================================================

input("ENTER ANY KEY")
