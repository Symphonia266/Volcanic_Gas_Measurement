# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from pyparsing import alphas
from scipy import constants as consts
from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file
from .consts import gases, s_raman_N2, s_raman_O2
from .atom import (
    N, 
    alphas_aer, 
    alphas_mol, 
    betas_aer, 
    betas_mol
)
from . import utils
from .lidar import Lidar
from .optics import overlap

from ..diffusion_model.diffuse_plume import DiffusePlumeLidar
from ..diffusion_model.func import gen_fauntainsource
from ..diffusion_model.pasquill_stable_classfication import (
    classification_label,
    classification_table,
    classify_atomosphere_stability,
    inverse_stab_class_to_wether,
)




# xs_SO2 = load_cross_section_dict(
#     {
#         "cold": "SO2_VandaeleHermansFally(2009)_298K_227.275-416.658nm.xlsx",
#         "hot":  "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
#     },
#     interp_kwargs={"bounds_error": False, "fill_value": np.nan},
# )

# xs_H2S = load_cross_section_dict(
#     {
#         "cold": "H2S_Grosch(2015)_294.8K_198-370nm.xlsx",
#         "hot":  "H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
#     },
#     interp_kwargs={"bounds_error": False, "fill_value": np.nan},
# )

xs_SO2 = utils.load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan}, 
    effective=
)
xs_H2S = utils.load_cross_section(
    "H2S_Grosch(2015)_423.2K_198-370nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan}
)
xs_O3 = utils.load_cross_section(
    "O3_Bogumil(2003)_293K_230-1070nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan}
)

E0:float=0.1
A:float=0.3
M: float = 100 * 60 * 60
eta: float = 0.3
q: float = 0.3
dR: float = 30
Bj: float = 0
F: float = 1
D: float = 0

@dataclass
class Gases:
    Q: float
    offset: float
    distribution: np.ndarray
class MeasurementModel:
    def __init__(
        self, windspeed, wind_direction, elevation, *, wether=None, stab_class=None
    ) -> None:
        self.speed = windspeed
        self.wind_direction = wind_direction
        self.diffusemodel = DiffusePlumeLidar(
            "pasquill", windspeed, wind_direction, wether=wether, stab_class=stab_class
        )
        self.wether = self.diffusemodel.wether
        self.stab_class = self.diffusemodel.stab_class

        self.lidar = Lidar(elevation=elevation)
        # self.wl = {"laser":np.arange(230, 370, 0.1)}
        # self.wl["N2_st"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], False)
        # self.wl["O2_st"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], False)
        # self.wl["N2_as"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], True)
        # self.wl["O2_as"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], True)

        self.wl = {"laser":300}
        self.wl["N2_st"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], False)
        self.wl["O2_st"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], False)
        self.wl["N2_as"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], True)
        self.wl["O2_as"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], True)

        self.gases = {}

        # self.entry_source(radius=5, cnt=(50, -40, 0), H=2)

    def entry_source(self, radius, cnt, H):
        x_src, y_src, z_src, q_src = gen_fauntainsource(radius=radius, cnt=cnt, N_pt=10)
        self.diffusemodel.entry_source(q_src, x_src, y_src, z_src, H)
        self.C = self.diffusemodel.Concentration(
            self.lidar.x_grid, z=self.lidar.z_grid, time_correction=10 * 60
        )

    def transmittance(self):
        xs_SO2 = {
            k:utils.calc_of_effective(self.wl[k], xs_SO2)
            for k in self.wl.keys()
        }
        xs_H2S = {
            k:utils.calc_of_effective(self.wl[k], xs_H2S)
            for k in self.wl.keys()
        }
        xs_O3 = {
            k:utils.calc_of_effective(self.wl[k], xs_O3)
            for k in self.wl.keys()
        }

    def entry_gases(self, name: str, Q:float, offset:float, xs):
        gas = Gases(Q=Q, offset=offset, distribution=Q * self.C + offset)
        self.gases[name] = gas

    def update_gases(self, *, name: str, Q=None, offset=None):
        if Q is not None:
            self.gases[name].Q = Q
        if offset is not None:
            self.gases[name].offset = offset

        self.gases[name].distribution = (
            self.gases[name].Q * 
            self.C + 
            self.gases[name].offset
        )

    def update_diffuse(
        self, *, windspeed=None, wind_direction=None, wether=None, stab_class=None
    ):
        self.diffusemodel.update_parameters(
            windspeed=windspeed,
            wind_direction=wind_direction,
            wether=wether,
            stab_class=stab_class,
        )
        self.C = self.diffusemodel.Concentration(
            self.lidar.x_grid, z=self.lidar.z_grid, time_correction=10 * 60
        )
        for gas in self.gases.keys():
            self.update_gases(name=gas)

    def show_gases(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(1,2,2, projection="3d")
        ax1.grid(which="minor", ls="--", c="lightgrey")
        ax1.grid(which="major", ls="-", c="darkgrey")
        ax3.grid(which="minor", ls="--", c="lightgrey")
        ax3.grid(which="major", ls="-", c="darkgrey")

        for name in self.gases.keys():
            ax1.scatter(self.lidar.distance, self.gases[name].distribution, clip_on=False,label=name)
        r = np.linspace(self.lidar.distance.min(), self.lidar.distance.max(), 1000)
        ax2.plot(
            r, self.diffusemodel.Concentration(r, time_correction=10 * 60), c="darkgrey",clip_on=False
        )
        ax1.set_xlabel("distance [m]")
        ax1.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
        ax1.legend()

        ax3.view_init(elev=30, azim=-110)
        x = np.linspace(self.lidar.x_grid.min(), self.lidar.x_grid.max(), 200)
        y = np.linspace(-50, 50, 200)
        x, y = np.meshgrid(x, y)
        C = self.diffusemodel.Concentration(x, y=y, z=0)
        ax3.plot_wireframe(
            x,
            y,
            C,
            color="blue",
            rstride=5,
            cstride=5,
            clip_on=False,
        )
        ax3.set_xlabel("X axis")
        ax3.set_ylabel("Y axis")
        ax3.set_zlabel("coefficient")
        ax1.set_ylim(0, 35)
        ax2.set_ylim(0, ax3.get_zlim()[1])
        # plt.tight_layout()
        plt.show(block=False)
        # input("ENTER ANY KEY...")

    def show_alphas(self):
        fig, axes = plt.subplots(1,2, layout="tight")
        for ax in axes:
            ax.grid(which="minor", ls="--", c="lightgrey")
            ax.grid(which="major", ls="-", c="darkgrey")
            ax.set_ylabel(r"absorptance $\alpha$")
        for key in self.alphas.keys():
            axes[0].scatter(self.lidar.distance, self.alphas[key][])        
            axes[1].scatter(self.wl["laser"], self.alphas[key])        

        plt.show(block=False)
        # input("ENTER ANY KEY...")
