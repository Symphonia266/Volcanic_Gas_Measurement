# coding: utf-8
import os
import sys
from gas_simulation.atom import alphas_mol
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gas_simulation.atom import alphas_aer, alphas_mol
from gas_simulation.model import Gas
from gas_simulation.model import Measurement as Measure
from gas_simulation import utils

samp = utils.load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
)

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

# 半径5mの真円煙源を有効煙源高度2mでセット
model = Measure(
    env_kwargs={
        # diffuse detail
        "windspeed": 2,
        "wind_direction": 90,
        "stab_class": "A",
        # source detail
        "source_kwargs": {
            "circ_args": {"radius": 5, "cnt": (50, -10), "N_pt": 25},
            "He": 1002,
        },
        # eruption gas details
        "gases": {
            "SO2": Gas(Q=4e4, offset=0.0, cross_section=xs_SO2),
            "H2S": Gas(Q=2e4, offset=0.0, cross_section=xs_H2S),
            "O3": Gas(Q=0, offset=0.0005, cross_section=xs_O3),
        },
    },
    lidar_kwargs={
        "end": 100,
        "elevation": 0,
        "alt_offset": 1000,
    },
)
model.show_gases()
tau = model.env.transmittance(model.lidar, 300)


# 煙源の水平位置を変えて再計算
model.set_parameter(
    source_kwargs={
        "circ_args": {"radius": 5, "cnt": (50, -50), "N_pt": 25},
        "He": 1002,
        "init": True,
    }
)
model.show_gases()
tau = model.env.transmittance(model.lidar, 300)

# # 煙流拡散条件を変えて再計算
# model.set_parameter(
#     diffuse_kwargs={"windspeed": 10, "wind_direction": 90, "wether": "overcast"},
#     gas_entry_dict={
#         "SO2": Gas(Q=8e4, offset=0.0, cross_section=xs_SO2),
#         "H2S": Gas(Q=4e4, offset=0.0, cross_section=xs_H2S),
#         "O3": Gas(Q=0, offset=0.0005, cross_section=xs_SO2),
#     },
# )
# print(model.env.diffuse.stab_class)
# model.show_gases()
# tau = model.env.transmittance(model.lidar, 300)

# # 点煙源に変えて再計算
# model.set_parameter(
#     diffuse_kwargs={
#         "wind_direction":90
#     },
#     source_kwargs={
#         "x_src":30,
#         "y_src":-40,
#         "He":2,
#         "init":True
#     },
# )
# model.show_gases()

input("ENTER ANY KEY......")
