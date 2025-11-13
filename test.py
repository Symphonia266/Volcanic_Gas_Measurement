# coding: utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gas_simulation.diffusion_model.diffuse_plume import DiffusePlumeLidar as DP
from gas_simulation.lidar_simulation.model import MeasurementModel as MM

windspeed = 10
stab_class = "C"
model = MM(windspeed=windspeed, wind_direction=90, elevation=0, stab_class=stab_class)
model.entry_gases("SO2", 10.0, 0.0)
model.entry_gases("H2S",  5.0, 0.0)
model.entry_gases("O3" ,  0.0, 0.005)
model.show_gases()
