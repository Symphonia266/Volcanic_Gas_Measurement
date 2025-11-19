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
    betas_N2, 
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
    effective=True
)
xs_H2S = utils.load_cross_section(
    "H2S_Grosch(2015)_423.2K_198-370nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True
)
xs_O3 = utils.load_cross_section(
    "O3_Bogumil(2003)_293K_230-1070nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True
)

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
        print(self.lidar.distance)
        print(self.lidar.x_grid)
        print(self.lidar.z_grid)

        # self.wl = {"laser":np.arange(230, 370, 0.1)}
        self.wl = {"laser":300}
        self.wl["N2_ST"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], False)
        self.wl["O2_ST"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], False)
        self.wl["N2_AS"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], True)
        self.wl["O2_AS"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], True)
 
        # self.wl_laser = np.arange(230, 370, 0.1)
        # wl = {
        #     "N2_ST" : utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], False),
        #     "O2_ST" : utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], False),
        #     "N2_AS" : utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], True),
        #     "O2_AS" : utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], True),
        # }
 
        # self.wl_on  = np.zeros_like(self.wl_laser)
        # self.wl_off = np.zeros_like(self.wl_laser)
        # a = xs_SO2(self.wl["N2_ST"])
        # b = xs_SO2(self.wl["O2_ST"])
        # flg = a>b
        # self.wl_on[flg]  = self.wl["N2_ST"][flg]
        # self.wl_off[flg] = self.wl["O2_ST"][flg]
        # self.wl_on[~flg]  = self.wl["O2_ST"][~flg]
        # self.wl_off[~flg] = self.wl["N2_ST"][~flg]

        self.gases = {}

        # self.entry_source(radius=5, cnt=(50, -40, 0), H=2)

    def entry_source(self, radius, cnt, He):
        x_src, y_src, q_src = gen_fauntainsource(radius=radius, cnt=cnt, N_pt=10)
        self.diffusemodel.entry_source(q_src, x_src, y_src, He)
        self.C = self.diffusemodel.Concentration(
            self.lidar.x_grid, z=self.lidar.z_grid, time_correction=10 * 60
        )

    def entry_gases(self, name: str, Q:float, offset:float):
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
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        ax3.set_axisbelow(True)
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

        ax3.view_init(elev=20, azim=-155)
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
        ax1.set_ylim(0, None)
        ax2.set_ylim(0, ax3.get_zlim()[1])
        # plt.tight_layout()
        plt.show(block=False)
        # input("ENTER ANY KEY...")

    def transmittance(self, scat1, scat2, dir1, dir2):
        feat = 1
        lb1 = f"{scat1}_{'AS' if dir1 else 'ST'}"
        lb2 = f"{scat2}_{'AS' if dir2 else 'ST'}"

        self.wl = {k:np.atleast_1d(v) for k, v in self.wl.items()}

        alpha_mol_laser = alphas_mol(self.wl["laser"][np.newaxis, :], self.lidar.z_grid[:, np.newaxis])
        alpha_aer_laser = alphas_aer(self.wl["laser"][np.newaxis, :], self.lidar.z_grid[:, np.newaxis], feat)

        alpha_mol_s1 = alphas_mol(self.wl[lb1][np.newaxis, :], self.lidar.z_grid[:, np.newaxis])
        alpha_aer_s1 = alphas_aer(self.wl[lb1][np.newaxis, :], self.lidar.z_grid[:, np.newaxis], feat)

        alpha_mol_s2 = alphas_mol(self.wl[lb2][np.newaxis, :], self.lidar.z_grid[:, np.newaxis])        
        alpha_aer_s2 = alphas_aer(self.wl[lb2][np.newaxis, :], self.lidar.z_grid[:, np.newaxis], feat)

        alpha_SO2_laser = self.gases["SO2"].distribution[:, np.newaxis] * xs_SO2(self.wl["laser"])[np.newaxis, :]
        alpha_SO2_s1 = self.gases["SO2"].distribution[:, np.newaxis] * xs_SO2(self.wl[lb1])[np.newaxis, :]
        alpha_SO2_s2 = self.gases["SO2"].distribution[:, np.newaxis] * xs_SO2(self.wl[lb2])[np.newaxis, :]

        alpha_H2S_laser = self.gases["H2S"].distribution[:, np.newaxis] * xs_H2S(self.wl["laser"])[np.newaxis, :]
        alpha_H2S_s1 = self.gases["H2S"].distribution[:, np.newaxis] * xs_H2S(self.wl[lb1])[np.newaxis, :]
        alpha_H2S_s2 = self.gases["H2S"].distribution[:, np.newaxis] * xs_H2S(self.wl[lb2])[np.newaxis, :]

        alpha_O3_laser = self.gases["O3"].distribution[:, np.newaxis] * xs_O3(self.wl["laser"])[np.newaxis, :]
        alpha_O3_s1 = self.gases["O3"].distribution[:, np.newaxis] * xs_O3(self.wl[lb1])[np.newaxis, :]
        alpha_O3_s2 = self.gases["O3"].distribution[:, np.newaxis] * xs_O3(self.wl[lb2])[np.newaxis, :]

        alpha_laser = alpha_mol_laser + alpha_aer_laser + alpha_SO2_laser + alpha_H2S_laser + alpha_O3_laser
        alpha_s1 = alpha_laser + alpha_mol_s1 + alpha_aer_s1 + alpha_SO2_s1 + alpha_H2S_s1 + alpha_O3_s1
        alpha_s2 = alpha_laser + alpha_mol_s2 + alpha_aer_s2 + alpha_SO2_s2 + alpha_H2S_s2 + alpha_O3_s2

        alpha_s1 = (alpha_s1[1:, :] + alpha_s1[:-1, :])*np.diff(self.lidar.distance)[:, np.newaxis]*0.5
        alpha_s2 = (alpha_s2[1:, :] + alpha_s2[:-1, :])*np.diff(self.lidar.distance)[:, np.newaxis]*0.5
        tau_scat1 = np.exp(-np.cumsum(alpha_s1, axis=0))
        tau_scat2 = np.exp(-np.cumsum(alpha_s2, axis=0))

        return tau_scat1, tau_scat2
