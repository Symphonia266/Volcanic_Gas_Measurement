# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from pathlib import Path
from dataclasses import dataclass
from typing import Protocol, TypeAlias
from matplotlib import pyplot as plt

# from scipy import constants as consts
# from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file

from . import utils
from .atom import alphas_aer, alphas_mol

from .diffusion_model.func import gen_fauntainsource
from .diffusion_model.diffuse_plume import Field
from .diffusion_model.diffuse_plume import Source
from .diffusion_model.diffuse_plume import PlumeModel
from .lidar_model.utils import Coord

__all__ = [
    "Field",
    "Source",
    "PlumeModel",
    "gen_fauntainsource",
    "Gas",
    "Environment", 
    "InstantEnvironment",
    "PlumeEnvironment",
]

Numeric: TypeAlias = float | np.ndarray

@dataclass
class Gas:
    Q: float
    offset: float
    cross_section: callable

class Environment(Protocol):
    time: float
    gas_inventory: dict[str, Gas]
    aer_absorp_feat: float

    def number_density_at(self, x, y, z)->dict[str, Numeric]:
        ...
    def transmittance(self, coord:Coord, wl:Numeric, *, axes=False)->Numeric:
        ...
    def plot_LoS_gases(self, ax, coord:Coord)->None:
        ...
    def show_gases(self, coord:Coord)->None:
        ...

class InstantEnvironment:
    def __init__(self, gas:dict[str, Gas], time: float, trig=None)->None:
        self.gas_inventory = {k: obj for k, obj in gas.items()}
        self.time = time
        self.aer_absorp_feat = 1.0

        if trig is None:
            self.trig = lambda x: False
        else:
            self.trig = trig

        print(f"elapsed time(and multiplier time)   : {utils.elapsed_time_str(time)}")
        print(f"multi. coeff. in aerzol absorptance : {self.aer_absorp_feat:.2f}\n")

    def number_density_at(self, x:Numeric, y:Numeric, z:Numeric)->dict[str, Numeric]:
        x = np.atleast_1d(x)
        n = np.zeros_like(x)
        n_gas = {}
        for key, obj in self.gas_inventory.items():
            mask = self.trig(x)
            n = np.where(mask, obj.Q, obj.offset)
            n_gas[key] = utils.ppm_to_number_density(n, z)
        return n_gas

    def transmittance(self, coord: Coord, wl: Numeric, *, axes= None)->Numeric:
        # ライダー視線原点の追加
        x_tmp, z_tmp = coord.get_xz(0)
        distance = np.append(0, coord.distance)
        x = np.append(x_tmp, coord.x)
        z = np.append(z_tmp, coord.z)

        # 距離・波長の二次元計算のための軸設定
        distance = np.atleast_1d(distance)[:, np.newaxis]
        x = np.atleast_1d(x)[:, np.newaxis]
        z = np.atleast_1d(z)[:, np.newaxis]
        wl = np.atleast_1d(wl)[np.newaxis, :]

        # 吸光度計算部
        absorptance_mol = alphas_mol(wl, z)
        absorptance_aer = alphas_aer(wl, z, self.aer_absorp_feat)
        n_gas = self.number_density_at(x, 0, z)
        absorptance_gas = {
            name: n_gas[name] * obj.cross_section(wl)
            for name, obj in self.gas_inventory.items()
        }
        absorptance = absorptance_mol + absorptance_aer + sum(absorptance_gas.values())

        # 光路累積加算部
        intgr = np.cumsum(
            (absorptance[:-1] + absorptance[1:]) * np.diff(distance, axis=0) * 0.5,
            axis=0,
        )
        transmittance = np.exp(-intgr)

        # 表示パネル提供
        if axes is not None:
            for ax in axes:
                ax.grid(which="major", ls="-", c="darkgrey")
                ax.grid(which="minor", ls="--", c="lightgrey")
                ax.set_xlabel("lidar distance [m]")
                # ax.set_yscale("log")
            axes[0].scatter(distance, absorptance[:, 0])
            axes[1].scatter(coord.distance, transmittance[:, 0])
            axes[0].set(
                ylabel="absorptance",
                ylim = (0, None)
            )
            axes[1].set(
                ylabel="transmittance",
                ylim=(0, 1)
            )
        return transmittance
    
    def plot_LoS_gases(self, ax:Axes, coord:Coord)->None:
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
        ax.set(
            xlabel="distance [m]",
            ylabel="concentration [ppm]",
            xlim = (coord.x.min(), coord.x.max()),
            ylim = (0, None),
        )
        ax.legend()
        # return ax

    def show_gases(self, coord: Coord)->None:
        fig, ax = plt.subplots(1, 1, layout="constrained")
        self.plot_LoS_gases(ax, coord)
        plt.show(block=False)

        # input("ENTER ANY KEY...")


class PlumeEnvironment(InstantEnvironment):
    def __init__(self, 
        gas: dict[str,Gas], 
        time: float, 
        field: Field, 
        source: Source
    )->None:
        super().__init__(gas, time)
        self.field = field
        self.source = source
        self.plume_model = PlumeModel(self.field, self.source)

    def number_density_at(self, x:Numeric, y:Numeric, z:Numeric)->dict[str, Numeric]:
        C = self.plume_model.calc(x, y, z, time=self.time)
        gas = {
            name: utils.ppm_to_number_density(obj.offset + obj.Q * (C), z)
            for name, obj in self.gas_inventory.items()
        }
        return gas

    def plot_diffuse_map_gases(self, ax:Axes, coord:Coord)->AxesImage:
        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.grid(which="major", ls="-", c="darkgrey")
        # r = np.linspace(coord.distance.min(), coord.distance.max(), 200)
        x = np.linspace(coord.x.min(), coord.x.max(), 200)
        y = np.linspace(-coord.x.max() / 2, coord.x.max() / 2, 200)
        # z = np.linspace(coord.z.min(), coord.z.max(), 200)
        C = self.plume_model.calc(
            x=x[np.newaxis, :],
            y=y[:, np.newaxis],
            z=coord.z0,
            time=self.time,
        )
        im = ax.imshow(
            C,
            origin="lower",  # 左下を (x_min, y_min) に
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="equal",
            cmap="jet",
        )
        ax.set(
            xlabel="X axis", 
            ylabel="Y axis"
        )
        return im

    def show_gases(self, coord: Coord)->None:
        fig, axes = plt.subplots(1,2, layout="constrained")
        self.plot_LoS_gases(axes[0], coord)
        im = self.plot_diffuse_map_gases(axes[1], coord)
        fig.colorbar(im, label="Concentration [units]")
        plt.show(block=False)
