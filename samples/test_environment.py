# coding: utf-8
import os
import sys
import numpy as np
from pathlib import Path

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from gas_simulation import utils
from gas_simulation.model import (
    Field,
    Source,
    gen_fauntainsource,
    Gas,
    PlumeEnvironment,
)
from gas_simulation.lidar_model.lidar import Lidar

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

# ライダーを高度1000mに設置（システムパラメータはデフォルトセッティング）
lidar = Lidar(end=100.0, alt_offset=1000)

# 大気状態プロファイルを設定
field = Field(windspeed=2, stab_class="A", wind_direction_deg=90)

# 半径5mの真円煙源を有効煙源高度2mでセット
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -10], N_pt=30)
He = np.full_like(q, 2 + lidar.alt_offset)
source = Source(Q=q, x=x_src, y=y_src, He=He)

# シミュレーション環境をセッティング
env = PlumeEnvironment(
    field,
    source,
    gas={
        "SO2": Gas(Q=30e5, offset=0, cross_section=xs_SO2),
        "H2S": Gas(Q=15e5, offset=0, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
    time=10 * 60,
)
env.show_gases(lidar)
tau = env.transmittance(lidar.distance, lidar.x_grid, lidar.z_grid, 300)

env.source.clear()
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[50, -21.5], N_pt=30)
He = np.full_like(q, 2 + lidar.alt_offset)
env.source.add(q, x_src, y_src, He)
env.show_gases(lidar)
tau = env.transmittance(lidar.distance, lidar.x_grid, lidar.z_grid, 300)


env.field.update(wind_direction_deg=0)
env.source.clear()
q, x_src, y_src = gen_fauntainsource(radius=5, cnt=[-10, 0], N_pt=30)
He = np.full_like(q, 2 + lidar.alt_offset)
env.source.add(q, x_src, y_src, He)
env.show_gases(lidar)
tau = env.transmittance(lidar.distance, lidar.x_grid, lidar.z_grid, 300)

env.field.update(windspeed=10, weather="overcast")
env.show_gases(lidar)
tau = env.transmittance(lidar.distance, lidar.x_grid, lidar.z_grid, 300)

input("ENTER ANY KEY......")
