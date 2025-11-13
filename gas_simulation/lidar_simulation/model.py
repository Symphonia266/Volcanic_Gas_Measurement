# coding: utf-8
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path

from . import package_path
from ..diffusion_model.diffuse_plume import DiffusePlumeLidar
from ..diffusion_model.func import gen_fauntainsource
from ..diffusion_model.pasquill_stable_classfication import (
  classification_label, 
  classification_table, 
  classify_atomosphere_stability, 
  inverse_stab_class_to_wether
)
class MeasurementModel:
    def __init__(self, windspeed, wind_direction, elevation, *, wether=None, stab_class=None) -> None:
        self.speed = windspeed
        self.wind_direction = wind_direction
        self.elevation = elevation
        self.diffusemodel = DiffusePlumeLidar("pasquill", windspeed, wind_direction, wether=wether, stab_class=stab_class)
        self.wether = self.diffusemodel.wether
        self.stab_class = self.diffusemodel.stab_class

        dR = 5
        self.distance = np.arange(0, 100, dR)
        self.x_grid = self.distance * np.cos(np.deg2rad(self.elevation))
        self.z_grid = self.distance * np.sin(np.deg2rad(self.elevation))
        self.gases = pd.DataFrame({"dist": self.distance})

        x_src, y_src, z_src, q_src = gen_fauntainsource(
            radius=5, 
            cnt=(50,-40,0),
            N_pt=100
        )
        H_src = 2
        self.diffusemodel.entry_source(q_src, x_src, y_src, z_src, H_src)
        self.C = self.diffusemodel.Concentration(self.x_grid, z=self.z_grid, time_correction=10*60)
    
    def entry_gases(self, gas_name:str, Q, offset):
        self.gases[gas_name] = Q*self.C + offset
    
    def show_gases(self):
        fig, ax = plt.subplots(1,1, layout="constrained")
        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.grid(which="major", ls="-", c="darkgrey")

        ax2 = ax.twinx()
        for key in self.gases.columns[1:]:
            ax.scatter(self.distance, self.gases[key], label=key)
        r = np.linspace(self.distance.min(), self.distance.max(), 1000)
        ax2.plot(r, self.diffusemodel.Concentration(r, time_correction=10*60), c="darkgrey")
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
        ax.legend()
        plt.show(block=False)
        input("ENTER ANY KEY...")
