# coding: utf-8
import os
import sys
from turtle import up
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt

# from scipy import constants as consts
# from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file

from . import utils
from .atom import alphas_aer, alphas_mol

from .lidar_model.lidar import Lidar

from .diffusion_model.func import gen_fauntainsource
from .diffusion_model.diffuse_plume import Field as BaseField
from .diffusion_model.diffuse_plume import Source as BaseSource
from .diffusion_model.diffuse_plume import PlumeModel
from gas_simulation.lidar_model import lidar

class Field(utils.Subject, BaseField):
    def __init__(
        self, 
        windspeed, *, 
        weather=None, 
        stab_class=None, 
        diffuse_model="pasquill", 
        wind_direction_deg=0
    ):
        super().__init__(
            windspeed, 
            weather=weather, 
            stab_class=stab_class, 
            diffuse_model=diffuse_model, 
            wind_direction_deg=wind_direction_deg
        )
    
    def update(self, *, 
            windspeed=None, 
            weather=None, 
            stab_class=None, 
            wind_direction_deg=None
        ):
        super().update(
            windspeed=windspeed, 
            weather=weather, 
            stab_class=stab_class, 
            wind_direction_deg=wind_direction_deg
        )
        self.notify()

class Source(utils.Subject, BaseSource):
    def __init__(self, Q=None, x=None, y=None, He=None):
        super().__init__(Q, x, y, He)

    def add(self, Q, x, y, He):
        super().add(Q, x, y, He)
        self.notify()

class Gas():
    def __init__(self, Q: float, offset: float, cross_section):
        self.Q = Q
        self.offset = offset
        self.cross_section = cross_section

    def concentration(self, C):
        return self.offset + self.Q * C
    
class Environment:
    def __init__(
            self, 
            field:Field,
            source:Source,
            lidar:Lidar,
            time:float,
            gas:dict
    ):
        self.field = field
        self.source = source
        self.plume_model = PlumeModel(self.field, self.source)

        self.gas_profile = {k:obj for k, obj in gas.items()}
        self.lidar = lidar

        self.time = time

        # 監視対象に登録
        self.field.attach(self)
        self.source.attach(self)
        self.aer_absorp_feat = 1

    def distribution(self, x, y, z):
        C = self.plume_model.concentration(x, y, z, time=self.time)
        gas = {name: utils.ppm_to_number_density(obj.concentration(C), z) for name, obj in self.gas_profile.items()}
        return gas

    def update(self, subject):
        """Field または Source から呼ばれる更新メソッド"""
        print(f"{subject} changed, recalculating plume...")
        self.gas = self.distribution(self.lidar.x_grid, 0, self.lidar.z_grid)

    def transmittance(self, wl):
        # calc all absorptance
        wl = np.atleast_1d(wl)

        absorptance_mol = alphas_mol(
            wl[np.newaxis, :], 
            self.lidar.z_grid[:, np.newaxis]
        )
        absorptance_aer = alphas_aer(
            wl[np.newaxis, :], 
            self.lidar.z_grid[:, np.newaxis], 
            self.aer_absorp_feat
        )
        n_gas = self.distribution(self.lidar.x_grid, 0, self.lidar.z_grid)
        absorptance_gas = {
            name : 
            n_gas[name][:, np.newaxis] * obj.cross_section(wl)[np.newaxis, :]
            for name, obj in self.gas_profile.items()
        }

        absorptance = absorptance_mol + absorptance_aer + sum(absorptance_gas.values())

        intgr = np.cumsum(
            (absorptance[:-1, :] + absorptance[1:, :])
            * np.diff(self.lidar.distance)[:, np.newaxis]
            * 0.5,
            axis=0,
        )
        transmittance = np.exp( -intgr )
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.grid(which="major", ls="-", c="darkgrey")
            ax.grid(which="minor", ls="--", c="lightgrey")
            ax.set_xlabel("lidar distance [m]")
            # ax.set_yscale("log")
        idx = np.searchsorted(wl, 300)
        axes[0].scatter(self.lidar.distance,     absorptance[:, idx])
        axes[1].scatter(self.lidar.distance[1:], transmittance[:, idx])
        axes[0].set_ylabel("absorptance")
        axes[1].set_ylabel("transmittance")
        axes[1].set_ylim(0, 1)
        plt.show(block=False)

        return transmittance
      
    def show_gases(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(1, 2, 2)
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        ax3.set_axisbelow(True)
        ax1.grid(which="minor", ls="--", c="lightgrey")
        ax1.grid(which="major", ls="-", c="darkgrey")
        ax3.grid(which="minor", ls="--", c="lightgrey")
        ax3.grid(which="major", ls="-", c="darkgrey")

        gas = self.distribution(self.lidar.x_grid, 0, self.lidar.z_grid)
        for name, n_gas in gas.items():
            ax1.scatter(
                self.lidar.distance,
                utils.number_density_to_ppm(n_gas, self.lidar.z_grid),
                clip_on=False,
                label=name,
           )
        ax2.plot(
            self.lidar.distance,
            self.plume_model.concentration(self.lidar.x_grid, 0, self.lidar.z_grid, time=self.time),
            c="darkgrey",
            clip_on=False,
        )
        ax1.set_xlabel("distance [m]")
        ax1.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
        ax1.legend()

        # ax3.view_init(elev=20, azim=-155)
        x = np.linspace(self.lidar.x_grid.min(), self.lidar.x_grid.max(), 200)
        y = np.linspace(-self.lidar.x_grid.max()/2, self.lidar.x_grid.max()/2, 200)
        C = self.plume_model.concentration(
            x=x[np.newaxis, :], 
            y=y[:, np.newaxis], 
            z=self.lidar.alt_offset, 
            time=self.time
        )
        im = ax3.imshow(
            C,
            origin='lower',      # 左下を (x_min, y_min) に
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect='equal',
            cmap='jet'
        )
        fig.colorbar(im, label='Concentration [units]')
        ax3.set_xlabel("X axis")
        ax3.set_ylabel("Y axis")
        # ax1.set_ylim(0, None)
        # ax2.set_ylim(0, ax3.get_zlim()[1])
        # plt.tight_layout()
        plt.show(block=False)
        # input("ENTER ANY KEY...")
