# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from diffusion_model.diffuse_plume import DiffusePlumeLidar as DP
from lidar_simulation.model import MeasurementModel as MM

windspeed = 10
stab_class = "C"
model = MM(windspeed=windspeed, azimuth=0, elevation=0, stab_class=stab_class)
model.entry_gases("SO2", 10.0, 0.0)
model.entry_gases("H2S",  5.0, 0.0)
model.entry_gases("O3" ,  0.0, 0.005)
model.show_gases()
