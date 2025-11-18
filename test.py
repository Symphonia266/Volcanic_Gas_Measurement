# coding: utf-8
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gas_simulation.diffusion_model.diffuse_plume import DiffusePlumeLidar as DP
from gas_simulation.lidar_simulation.model import MeasurementModel as MM

windspeed = 2
stab_class = "A"
model = MM(windspeed=windspeed, wind_direction=-45, elevation=0, stab_class=stab_class)
model.entry_source(radius=5,  cnt=(20, 0, 0), H=2)
model.entry_gases("SO2", 2.5 * 1e4, 0.0)
model.entry_gases("H2S", 1.0 * 1e4, 0.0)
model.entry_gases("O3", 0.0, 0.005)
model.show_gases()

model.diffusemodel.clear_source()
model.entry_source(radius=5, cnt=(50, 40, 0), H=2)
model.update_diffuse(windspeed=windspeed, wind_direction=-90, stab_class=stab_class)
model.show_gases()

input("ENTER ANY KEY......")
