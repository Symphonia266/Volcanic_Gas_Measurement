# coding: utf-8
from dataclasses import dataclass
import sys
from turtle import mode
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
    inverse_stab_class_to_wether,
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
        self.elevation = elevation
        self.diffusemodel = DiffusePlumeLidar(
            "pasquill", windspeed, wind_direction, wether=wether, stab_class=stab_class
        )
        self.wether = self.diffusemodel.wether
        self.stab_class = self.diffusemodel.stab_class

        dR = 5
        self.distance = np.arange(0, 100, dR)
        self.x_grid = self.distance * np.cos(np.deg2rad(self.elevation))
        self.z_grid = self.distance * np.sin(np.deg2rad(self.elevation))

        self.gases = {}

        self.entry_source(radius=5, cnt=(50, -40, 0), H=2)
        self.C = self.diffusemodel.Concentration(
            self.x_grid, z=self.z_grid, time_correction=10 * 60
        )

    def entry_source(self, radius, cnt, H):
        x_src, y_src, z_src, q_src = gen_fauntainsource(radius=radius, cnt=cnt, N_pt=10)
        self.diffusemodel.entry_source(q_src, x_src, y_src, z_src, H)
        self.C = self.diffusemodel.Concentration(
            self.x_grid, z=self.z_grid, time_correction=10 * 60
        )

    def entry_gases(self, name: str, Q, offset):
        self.gases[name] = Gases(Q=Q, offset=offset, distribution=Q * self.C + offset)

    def update_gases(self, *, name: str, Q=None, offset=None):
        if Q is not None:
            self.gases[name].Q = Q
        if offset is not None:
            self.gases[name].offset = offset
        self.gases[name].distribution = (
            Q if Q is not None else self.gases[name].Q
        ) * self.C + (offset if offset is not None else self.gases[name].offset)

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
            self.x_grid, z=self.z_grid, time_correction=10 * 60
        )
        for gas in self.gases.keys():
            self.update_gases(name=gas)

    def show_gases(self):
        fig, ax = plt.subplots(1, 1, layout="constrained")

        ax.grid(which="minor", ls="--", c="lightgrey")
        ax.grid(which="major", ls="-", c="darkgrey")

        ax2 = ax.twinx()
        for key in self.gases.keys():
            ax.scatter(self.distance, self.gases[key].distribution, label=key)
        r = np.linspace(self.distance.min(), self.distance.max(), 1000)
        ax2.plot(
            r, self.diffusemodel.Concentration(r, time_correction=10 * 60), c="darkgrey"
        )
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
        ax.legend()

        ax3.view_init(elev=15, azim=20)
        x = np.linspace(self.x_grid.min(), self.x_grid.max(), 200)
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
        )
        ax3.set_xlabel("X axis")
        ax3.set_ylabel("Y axis")

        plt.show(block=False)
        input("ENTER ANY KEY...")
