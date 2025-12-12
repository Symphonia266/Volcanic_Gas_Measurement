# coding: utf-8
from calendar import c
import os
from re import X
import sys
from turtle import distance
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt

# from scipy import constants as consts
# from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file

from . import utils
from .atom import alphas_aer, alphas_mol

from .lidar_model.lidar import Lidar

from .diffusion_model.func import gen_fauntainsource
from .diffusion_model.diffuse_plume import Field
from .diffusion_model.diffuse_plume import Source
from .diffusion_model.diffuse_plume import PlumeModel
from .lidar_model.lidar import Coord, Lidar

__all__ = [
    "Field",
    "Source",
    "PlumeModel",
    "gen_fauntainsource",
    "Gas",
    "PlumeEnvironment",
]


@dataclass
class Gas:
    Q: float
    offset: float
    cross_section: callable


class InstantEnvironment:
    def __init__(self, gas: dict, time: float, trig: callable = None):

        self.gas_inventory = {k: obj for k, obj in gas.items()}
        self.time = time
        self.aer_absorp_feat = 1.0

        if trig is None:
            self.trig = lambda x: False

        else:
            self.trig = trig

        print(f"elapsed time(and multiplier time)   : {utils.elapsed_time_str(time)}")
        print(f"multi. coeff. in aerzol absorptance : {self.aer_absorp_feat:.2f}\n")

    def number_density_at(self, x, y, z):
        x = np.atleast_1d(x)
        n = np.zeros_like(x)
        n_gas = {}
        for key, obj in self.gas_inventory.items():
            mask = self.trig(x)
            n = np.where(mask, obj.Q, obj.offset)
            n_gas[key] = utils.ppm_to_number_density(n, z)
        return n_gas

    def transmittance(self, coord: Coord, wl: float | np.ndarray, *, show=False):
        x_tmp, z_tmp = coord.get_xz(0)
        distance = np.append(0, coord.distance)
        x = np.append(x_tmp, coord.x)
        z = np.append(z_tmp, coord.z)

        distance = np.atleast_1d(distance)[:, np.newaxis]
        x = np.atleast_1d(x)[:, np.newaxis]
        z = np.atleast_1d(z)[:, np.newaxis]
        wl = np.atleast_1d(wl)[np.newaxis, :]

        absorptance_mol = alphas_mol(wl, z)
        absorptance_aer = alphas_aer(wl, z, self.aer_absorp_feat)
        n_gas = self.number_density_at(x, 0, z)
        # for name, obj in self.gas_inventory.items():
        #     a = n_gas[name]
        #     b = obj.cross_section(wl)
        #     print(f"{name:>5}: n_gas shape: {a.shape}, cross-sec shape: {b.shape}")
        absorptance_gas = {
            name: n_gas[name] * obj.cross_section(wl)
            for name, obj in self.gas_inventory.items()
        }

        absorptance = absorptance_mol + absorptance_aer + sum(absorptance_gas.values())

        intgr = np.cumsum(
            (absorptance[:-1] + absorptance[1:]) * np.diff(distance, axis=0) * 0.5,
            axis=0,
        )
        transmittance = np.exp(-intgr)
        if show:
            fig, axes = plt.subplots(1, 2)
            for ax in axes:
                ax.grid(which="major", ls="-", c="darkgrey")
                ax.grid(which="minor", ls="--", c="lightgrey")
                ax.set_xlabel("lidar distance [m]")
                # ax.set_yscale("log")
            axes[0].scatter(distance, absorptance[:, 0])
            axes[1].scatter(coord.distance, transmittance[:, 0])
            axes[0].set_ylabel("absorptance")
            axes[1].set_ylabel("transmittance")
            axes[0].set_ylim(0, None)
            axes[1].set_ylim(0, 1)
            plt.show(block=False)
        return transmittance

    def show_gases(self, coord: Coord):
        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.grid(which="major", ls="-", c="darkgrey")

        gas = self.number_density_at(coord.x, 0, coord.z)
        for name, n in gas.items():
            ax.plot(
                coord.distance,
                utils.number_density_to_ppm(n, coord.z),
                marker="o",
                label=name,
            )
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("concentration [ppm]")
        ax.set_xlim(coord.x.min(), coord.x.max())
        ax.set_ylim(0, None)
        ax.legend()

        plt.show(block=False)
        # input("ENTER ANY KEY...")


class PlumeEnvironment(InstantEnvironment):
    def __init__(self, gas: dict, time: float, field: Field, source: Source):
        super().__init__(gas, time)
        self.field = field
        self.source = source
        self.plume_model = PlumeModel(self.field, self.source)

    def number_density_at(self, x, y, z):
        C = self.plume_model.calc(x, y, z, time=self.time)
        gas = {
            name: utils.ppm_to_number_density(obj.offset + obj.Q * (C), z)
            for name, obj in self.gas_inventory.items()
        }
        return gas

    def show_gases(self, coord: Coord):
        r = np.linspace(coord.distance.min(), coord.distance.max(), 200)
        x = np.linspace(coord.x.min(), coord.x.max(), 200)
        z = np.linspace(coord.z.min(), coord.z.max(), 200)

        fig = plt.figure(layout="constrained")
        ax1 = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2)
        ax1.set_axisbelow(True)
        ax3.set_axisbelow(True)
        ax1.grid(which="minor", ls="--", c="lightgrey")
        ax1.grid(which="major", ls="-", c="darkgrey")
        ax3.grid(which="minor", ls="--", c="lightgrey")
        ax3.grid(which="major", ls="-", c="darkgrey")
        gas = self.number_density_at(coord.x, 0, coord.z)
        for name, n in gas.items():
            ax1.plot(
                coord.distance,
                utils.number_density_to_ppm(n, coord.z),
                marker="o",
                label=name,
            )
        ax1.set_xlabel("distance [m]")
        ax1.set_ylabel("concentration [ppm]")
        ax1.set_xlim(coord.x.min(), coord.x.max())
        ax1.set_ylim(0, None)
        ax1.legend()

        # ax3.view_init(elev=20, azim=-155)
        y = np.linspace(-coord.x.max() / 2, coord.x.max() / 2, 200)
        C = self.plume_model.calc(
            x=x[np.newaxis, :],
            y=y[:, np.newaxis],
            z=coord.z0,
            time=self.time,
        )
        im = ax3.imshow(
            C,
            origin="lower",  # 左下を (x_min, y_min) に
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="equal",
            cmap="jet",
        )
        fig.colorbar(im, label="Concentration [units]")
        ax3.set_xlabel("X axis")
        ax3.set_ylabel("Y axis")
        plt.show(block=False)
