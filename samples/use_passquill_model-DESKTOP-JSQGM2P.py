from cProfile import label
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from diffusion_model.pasquill_stable_classfication import PGStableClassfication as PG
from diffusion_model.pasquill_model.pasquill_gifford_spreadwidth import SpreadWidth as PasquillSpread
from diffusion_model.sutton_model.sutton_spreadwidth import SpreadWidth as SuttonSpread
windspeed = 5
wether = "clear"
atom_model = PG(windspeed=windspeed, wether=wether, stab_class=None)
"""
これは現在の風速, 日照（＝天気）から, パスキルの大気安定度分類に基づいて安定分類を行うクラスです. 
所属メソッドとして安定度分類更新のためのupdate_stability_classfiedが存在します. 
これに連なるパスキルの拡散幅モデル, サットンの拡散幅モデルに対して最も基幹的なクラスになります. 
stab_classは風速, 天気に関わらず任意の分類を導入するためのテスト用引数です. 
"""

print(f"Atomospheric stability is : {atom_model.stability_class}")
atom_model.update_stability_classfied(windspeed=windspeed+1, wether="overcast", stab_class=None)

spread_model = PasquillSpread()
"""
これは点源から湧出する気体が一定の風速・風向によって形成する煙流の鉛直・水平方向の拡散幅を導出するクラスです. 
所属するメソッドとして, 水平方向の拡散幅を返すlateral, 鉛直方向の拡散幅を返すverticalが存在します.
これは安定分類クラスを継承しており, 安定度が高いほど拡散幅は水平方向・鉛直方向共に減少する傾向があります. 
拡散幅はPasquill-Gifford線図を手動でサンプリングしたプロットデータから
近傍, 遠方のデータを切り取ってlog-log線形回帰による外挿を施すことで
100mから100,000m区間のデータしかなかった原著から拡張しています. 
また, 外挿関数部とサンプリングデータの線形補完部の接続は対数重みによるスムージングを行っています. 
"""
import numpy as np
import matplotlib.pyplot as plt

i = np.linspace(0, 6)
x = np.power(10, i)


fig, axes = plt.subplots(1,2)
for ax in axes.ravel():
  ax.grid(which="major", ls="-", c="darkgrey")
  ax.grid(which="minor", ls="--", c="lightgrey")
  ax.set_xlabel("downwind distance [m]")
  ax.set_xscale("log")
  ax.set_yscale("log")

spread_y = {}; spread_z={}
for lb in PG.lbs:
  spread_y[lb] = spread_model.lateral(x, lb)
  spread_z[lb] = spread_model.vertical(x, lb)
  axes[0].plot(x, spread_y[lb], label=lb)
  axes[1].plot(x, spread_z[lb], label=lb)
  axes[0].set_ylabel("lateral spread width [m / 3 min]")
  axes[1].set_ylabel("vertical spread width [m / 3 min]")
axes[0].legend()
plt.show()
