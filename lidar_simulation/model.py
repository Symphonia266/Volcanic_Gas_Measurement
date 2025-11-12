# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from . import package_path
from ..diffusion_model.diffuse_plume import DiffusePlumeLidar
from ..diffusion_model.pasquill_stable_classfication import classification_label, classification_table, classify_atomosphere_stability

class MeasurementModel:
    def __init__(self, windspeed, azimuth, elevation, wether, stab_class) -> None:
        self.speed = windspeed
        self.azimuth = azimuth
        self.elevation = elevation
        if stab_class == None:
            # self.wether = wether
            self.stab_class = classify_atomosphere_stability(windspeed, wether)
        else:
            self.stab_class = stab_class
        self.diffusemodel = DiffusePlumeLidar("pasquill", windspeed, azimuth, elevation, None, self.stab_class)

        end = 1000
        dR = 5
        self.lidar_grid = np.arange(0, end, dR)
        