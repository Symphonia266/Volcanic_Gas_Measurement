# coding: utf-8
import os
import sys
from types import LambdaType
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
# from scipy import constants as consts
# from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file

from . import utils
from .consts import gases, s_raman_N2, s_raman_O2
from .atom import N, alphas_aer, alphas_mol, betas_N2, betas_aer, betas_mol

from .lidar_model.lidar import Lidar

from .diffusion_model.pasquill_stable_classfication import (
    classification_label,
    classification_table,
    classify_atomosphere_stability,
    inverse_stab_class_to_wether,
)
from .diffusion_model.func import gen_fauntainsource
from .diffusion_model.diffuse_plume import DiffusePlumeLidar

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

def gen_conbi(wl, scat1, scat2, dir1, dir2):
    wl_s1 = utils.wl_shift(wl, gases.at[scat1, "sft"], dir1)
    wl_s2 = utils.wl_shift(wl, gases.at[scat2, "sft"], dir2)
    return wl_s1, wl_s2

class Gas:
    def __init__(self, Q:float, offset:float, cross_section):
        self.Q = Q
        self.offset = offset
        self.cross_section = cross_section
        self.concentration: np.ndarray
class Enviroment:
    def __init__(
        self, 
        windspeed, 
        wind_direction, 
        source_kwargs:dict, 
        gases:dict,
        *, 
        wether=None, 
        stab_class=None

    ) -> None:
        self.speed = windspeed
        self.wind_direction = wind_direction
        self.diffuse = DiffusePlumeLidar(
            windspeed, wind_direction, wether=wether, stab_class=stab_class, model="pasquill"
        )
        self.set_source(**source_kwargs)
        self.gases = {k: v for k, v in gases.items()}
        self.aer_absorp_feat = 1
        
    def set_source(
        self, 
        *,
        circ_args:dict|None=None, 
        x_src=None, 
        y_src=None, 
        q_src=1, 
        He=None,
        init=False
    ):
        if init:
            self.diffuse.clear_source()
        if circ_args is not None:
            q_src, x_src, y_src = gen_fauntainsource(**circ_args)
        
        if (x_src is None) or (y_src is None) or (He is None):
            raise ValueError('source "x" or "y" or "He" is not defined.')

        self.diffuse.entry_source(q_src, x_src, y_src, He)

    def transmittance(self, lidar, wl):
        # calc all absorptance
        wl = np.atleast_1d(wl)
        
        absorptance_mol = alphas_mol(wl[np.newaxis, :], lidar.z_grid[:, np.newaxis])
        absorptance_aer = alphas_aer(wl[np.newaxis, :], lidar.z_grid[:, np.newaxis], self.aer_absorp_feat)
        absorptance_gas = {
            name : obj.concentration[:, np.newaxis] * obj.cross_section(wl[np.newaxis, :]) 
            for name, obj in self.gases.items()
        }
        absorptance = (
                absorptance_mol
                +absorptance_aer
                +sum(absorptance_gas.values())
        )

        intgr = np.cumsum((absorptance[:-1, :]+absorptance[1:, :])*np.diff(lidar.distance)[:, np.newaxis]*0.5, axis=0)
        transmittance = np.exp(-intgr)

        fig, axes = plt.subplots(1,2)
        for ax in axes:
            ax.grid(which="major", ls="-", c="darkgrey")
            ax.grid(which="minor", ls="--", c="lightgrey")
            ax.set_xlabel("lidar distance [m]")
            # ax.set_yscale("log")
        
        axes[0].scatter(lidar.distance, absorptance.ravel())
        axes[1].scatter(lidar.distance[1:], transmittance.ravel())
        axes[0].set_ylabel("absorptance")
        axes[1].set_ylabel("transmittance")
        plt.show(block=False)

        return transmittance

class Measurement:
    def __init__(
        self, 
        env_kwargs, 
        lidar_kwargs
    ):
        env_kwargs = dict(env_kwargs or {})
        lidar_kwargs = dict(lidar_kwargs or {})

        self.env = Enviroment(**env_kwargs)
        self.lidar = Lidar(**lidar_kwargs)

        self.C = self.env.diffuse.Concentration(
            self.lidar.x_grid, z=self.lidar.z_grid, time_correction=10 * 60
        )
        for name, obj in self.env.gases.items():
            obj.concentration = (obj.offset + obj.Q*self.C)*1e-6*N(self.lidar.z_grid)
        
        # =============================================================================
        self.wl = {"laser": 300}
        self.wl["N2_st"] = utils.wl_shift(
            self.wl["laser"], gases.at["N2", "sft"], False
        )
        self.wl["O2_st"] = utils.wl_shift(
            self.wl["laser"], gases.at["O2", "sft"], False
        )
        self.wl["N2_as"] = utils.wl_shift(self.wl["laser"], gases.at["N2", "sft"], True)
        self.wl["O2_as"] = utils.wl_shift(self.wl["laser"], gases.at["O2", "sft"], True)
        # ==============================================================================

    def set_parameter(
            self,
            *, 
            diffuse_kwargs=None,
            source_kwargs=None,
            gas_entry_dict:dict|None=None
    ):
        if diffuse_kwargs is not None:
            diffuse_kwargs = dict(diffuse_kwargs or {})
            self.env.diffuse.update_parameters(**diffuse_kwargs)

        if source_kwargs is not None:
            source_kwargs = dict(source_kwargs or {})
            self.env.set_source(**source_kwargs)

        if gas_entry_dict is not None:
            for k, v in gas_entry_dict.items():
                self.gases = {k: v for k, v in gas_entry_dict.items()}

        self.C = self.env.diffuse.Concentration(
            self.lidar.x_grid, 
            z=self.lidar.z_grid, 
            time_correction=10 * 60
        )
        for name, obj in self.env.gases.items():
            obj.concentration = obj.offset + obj.Q*self.C

    def show_gases(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        ax3.set_axisbelow(True)
        ax1.grid(which="minor", ls="--", c="lightgrey")
        ax1.grid(which="major", ls="-", c="darkgrey")
        ax3.grid(which="minor", ls="--", c="lightgrey")
        ax3.grid(which="major", ls="-", c="darkgrey")

        for key, obj in self.env.gases.items():
            ax1.scatter(
                self.lidar.distance,
                obj.concentration*1e6/N(self.lidar.z_grid),
                clip_on=False,
                label=key,
            )
        r = np.linspace(self.lidar.distance.min(), self.lidar.distance.max(), 1000)
        ax2.plot(
            r,
            self.env.diffuse.Concentration(r, z=self.lidar.alt_offset, time_correction=10 * 60),
            c="darkgrey",
            clip_on=False,
        )
        ax1.set_xlabel("distance [m]")
        ax1.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
        ax1.legend()

        ax3.view_init(elev=20, azim=-155)
        x = np.linspace(self.lidar.x_grid.min(), self.lidar.x_grid.max(), 200)
        y = np.linspace(-50, 50, 200)
        x, y = np.meshgrid(x, y)
        C = self.env.diffuse.Concentration(x, y=y, z=self.lidar.alt_offset)
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
