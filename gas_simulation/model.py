# coding: utf-8
import os
from re import X
import sys
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
from .lidar_model.lidar import rotate, Lidar

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


class PlumeEnvironment:
    def __init__(
        self, field: Field, source: Source, time: float, gas: dict
    ):
        self.field = field
        self.source = source
        self.plume_model = PlumeModel(self.field, self.source)

        self.gas_inventory = {k: obj for k, obj in gas.items()}
        self.time = time
        self.aer_absorp_feat = 1

    def number_density_at(self, x, y, z):
        C = self.plume_model.calc(x, y, z, time=self.time)
        gas = {
            name: utils.ppm_to_number_density(obj.offset + obj.Q * (C), z)
            for name, obj in self.gas_inventory.items()
        }
        return gas

    def transmittance(self, distance, x, z, wl):
        # calc all absorptance
        wl = np.atleast_1d(wl)

        absorptance_mol = alphas_mol(
            wl[np.newaxis, :], z[:, np.newaxis]
        )
        absorptance_aer = alphas_aer(
            wl[np.newaxis, :], z[:, np.newaxis], self.aer_absorp_feat
        )
        n_gas = self.number_density_at(x, 0, z)
        absorptance_gas = {
            name: n_gas[name][:, np.newaxis] * obj.cross_section(wl)[np.newaxis, :]
            for name, obj in self.gas_inventory.items()
        }

        absorptance = absorptance_mol + absorptance_aer + sum(absorptance_gas.values())

        intgr = np.cumsum(
            (absorptance[:-1, :] + absorptance[1:, :])
            * np.diff(distance)[:, np.newaxis]
            * 0.5,
            axis=0,
        )
        transmittance = np.exp(-intgr)
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.grid(which="major", ls="-", c="darkgrey")
            ax.grid(which="minor", ls="--", c="lightgrey")
            ax.set_xlabel("lidar distance [m]")
            # ax.set_yscale("log")
        idx = np.searchsorted(wl, 300)
        axes[0].scatter(distance,       absorptance[:, idx])
        axes[1].scatter(distance[1:],   transmittance[:, idx])
        axes[0].set_ylabel("absorptance")
        axes[1].set_ylabel("transmittance")
        axes[0].set_ylim(0, None)
        axes[1].set_ylim(0, 1)
        plt.show(block=False)

        return transmittance

    def show_gases(self, lidar):
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

        gas = self.number_density_at(lidar.x_grid, 0, lidar.z_grid)
        p1 = []
        for name, n in gas.items():
            p = ax1.scatter(
                lidar.distance,
                utils.number_density_to_ppm(n, lidar.z_grid),
                clip_on=False,
                label=name,
            )
            p1.append(p)
        r = np.linspace(lidar.distance.min(), lidar.distance.max(), 200)
        x = np.linspace(lidar.x_grid.min(),   lidar.x_grid.max(), 200)
        z = np.linspace(lidar.z_grid.min(),   lidar.z_grid.max(), 200)
        C = self.plume_model.calc(x, 0, z, time=self.time)
        p2 = ax2.plot(r, C / C.max(), c="darkgrey", clip_on=False, label="plume-coeff.")
        ax1.set_xlabel("distance [m]")
        ax1.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient(normalized)")
        ax1.set_ylim(0, None)
        ax2.set_ylim(0, 1)
        ax1.legend(handles=[*p1, *p2])

        # ax3.view_init(elev=20, azim=-155)
        y = np.linspace(-lidar.x_grid.max() / 2, lidar.x_grid.max() / 2, 200)
        C = self.plume_model.calc(
            x=x[np.newaxis, :],
            y=y[:, np.newaxis],
            z=lidar.alt_offset,
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
        # ax1.set_ylim(0, None)
        # ax2.set_ylim(0, ax3.get_zlim()[1])
        plt.tight_layout()
        plt.show(block=False)
        # input("ENTER ANY KEY...")
