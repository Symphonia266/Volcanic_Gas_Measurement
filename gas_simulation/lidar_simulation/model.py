# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file
from ..diffusion_model.diffuse_plume import DiffusePlumeLidar
from ..diffusion_model.func import gen_fauntainsource
from ..diffusion_model.pasquill_stable_classfication import (
    classification_label,
    classification_table,
    classify_atomosphere_stability,
    inverse_stab_class_to_wether,
)

def _read_excel_safe(fname, **kwargs):
    p = data_dir / fname
    if not p.exists():
        raise FileNotFoundError(f"file are not found: {p}")
    return pd.read_excel(p, **kwargs)

def load_cross_section(fname, *, skiprows=5, wl_col=0, xs_col=1, interp_kwargs=None, **read_kwargs):
    """
    Excel ファイルを読み、WL/XS 列を取り出して interp1d を返す。
    - skiprows: ヘッダ等のスキップ行数（既定値は 5）
    - interp_kwargs: interp1d に渡す dict（例: {'bounds_error':False, 'fill_value':'extrapolate'}）
    """
    interp_kwargs = dict(interp_kwargs or {})
    df = _read_excel_safe(
        fname, 
        skiprows=skiprows, 
        header=None, 
        usecols=[wl_col, xs_col], 
        names=["WL", "XS"], 
        **read_kwargs
    )
    df = df.dropna(subset=["WL", "XS"]).sort_values("WL")
    wl = df["WL"].to_numpy()
    xs = df["XS"].to_numpy()
    return interp1d(wl, xs, **interp_kwargs)

def load_cross_section_dict(mapping, *, interp_kwargs=None, **read_kwargs):
    return {
        k: load_cross_section(v, interp_kwargs=interp_kwargs, **read_kwargs) 
        for k, v in mapping.items()
    }

xs_SO2 = load_cross_section_dict(
    {
        "cold": "SO2_VandaeleHermansFally(2009)_298K_227.275-416.658nm.xlsx",
        "hot":  "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    },
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
)

xs_H2S = load_cross_section_dict(
    {
        "cold": "H2S_Grosch(2015)_294.8K_198-370nm.xlsx",
        "hot":  "H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
    },
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
)

xs_O3 = load_cross_section(
    "O3_Bogumil(2003)_293K_230-1070nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan})


print(xs_SO2["hot"](230))
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
        self.z_grid = self.distance * np.sin(np.deg2rad(self.elevation))+1

        self.gases = {}


        # self.entry_source(radius=5, cnt=(50, -40, 0), H=2)

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
            self.x_grid, z=self.z_grid, time_correction=10 * 60
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
            ax1.scatter(self.distance, self.gases[name].distribution, clip_on=False,label=name)
        r = np.linspace(self.distance.min(), self.distance.max(), 1000)
        ax2.plot(
            r, self.diffusemodel.Concentration(r, time_correction=10 * 60), c="darkgrey",clip_on=False
        )
        ax1.set_xlabel("distance [m]")
        ax1.set_ylabel("concentration [ppm]")
        ax2.set_ylabel("coefficient")
        ax1.legend()

        ax3.view_init(elev=30, azim=-110)
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
