# coding: utf-8
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from . import package_path
from ..diffusion_model.diffuse_plume import DiffusePlumeLidar
from ..diffusion_model.pasquill_stable_classfication import (
  classification_label, 
  classification_table, 
  classify_atomosphere_stability, 
  inverse_stab_class_to_wether
)
class MeasurementModel:
    def __init__(self, windspeed, azimuth, elevation, *, wether=None, stab_class=None) -> None:
        self.speed = windspeed
        self.azimuth = azimuth
        self.elevation = elevation
        self.diffusemodel = DiffusePlumeLidar("pasquill", windspeed, azimuth, elevation, wether=wether, stab_class=stab_class)
        self.wether = self.diffusemodel.core.wether
        self.stab_class = self.diffusemodel.core.stab_class

        end = 1000
        dR = 5
        self.lidar_grid = np.arange(0, end, dR)
        self.gases = pd.DataFrame({"dist": self.lidar_grid})
        self.C = self.diffusemodel.Concentration(x_lidar=self.lidar_grid, time_correction=10*60)
    
    def entry_gases(self, gas_name:str, Q, offset):
        self.gases[gas_name] = Q*self.C + offset
    
    def show_gases(self):
        fig, ax = plt.subplots(1,1, layout="constrained")
        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.grid(which="major", ls="-", c="darkgrey")
        ax2 = ax.twinx()
        for key in self.gases.columns:
            ax.scatter(self.lidar_grid, self.gases[key], label=key)
        x = np.linspace(self.lidar_grid.min(), self.lidar_grid.max(), 1000)
        ax2.plot(x, self.diffusemodel.Concentration(x_lidar=x, time_correction=10*60), lc="darkgrey")
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
