# coding: utf-8
import numpy as np
from tqdm import tqdm

from .pasquill_stable_classfication import (
    classify_atomosphere_stability,
    inverse_stab_class_to_wether,
)
from .pasquill_model.pasquill_gifford_spreadwidth import SpreadWidth as PasquillSpread
from .sutton_model.sutton_spreadwidth import SpreadWidth as SuttonSpread
from .func import correct_time


def plume_kernel(
    x, 
    y, 
    z, 
    q,
    He, 
    model:PasquillSpread, 
    stab_class:str, 
    windspeed:float,
    *,
    time:float|None=None
):
    """
    Gaussian plume calculation kernel for multiple sources (vectorized).
    
    Parameters
    ----------
    x, y, z : array-like
        Grid coordinates, can be broadcastable shapes.
        Last dimension corresponds to sources.
    Q : array-like
        Source emission rates (same length as number of sources).
    He : array-like
        Effective source heights (same length as number of sources).
    model : PasquillSpread
        Lateral/vertical spread model.
    stab_class : str
        Atmospheric stability class.
    windspeed : float
        Wind speed at source.
    time : float, optional
        Time correction factor. Defaults to None.
    
    Returns
    -------
    C : np.ndarray
        Concentration at grid points (same shape as x, y, z excluding sources dimension)
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x, y, z = np.broadcast_arrays(x, y, z)
    C = np.zeros_like(x, dtype=float)

            
    mask = x > 0
    sigma_y = model.lateral(x[mask], stab_class)
    sigma_z = model.vertical(x[mask], stab_class)

    # --- windspeed correction ---
    if time is not None:
        windspeed = windspeed * time
        sigma_y = correct_time(time, sigma_y)
        sigma_z = correct_time(time, sigma_z)
    else: windspeed = windspeed * 3 * 60
    
    if mask.any():
        C[mask] = (
            q / 
            (2 * np.pi * sigma_y * sigma_z * windspeed) * 
            np.exp(-(y[mask] ** 2) / (2 * sigma_y**2)) * 
            (
                np.exp(-((z[mask] - He) ** 2) / (2 * sigma_z**2)) +
                np.exp(-((z[mask] + He) ** 2) / (2 * sigma_z**2))
            )
        )
    return C

class Field:
    """
    Represents an atmospheric field with wind speed, stability, and spread model.
    """
    def __init__(
            self, 
            windspeed, 
            *, 
            weather=None, 
            stab_class=None, 
            diffuse_model="pasquill"
    ):
        self.windspeed = windspeed

        if      diffuse_model == "pasquill" : self.spread = PasquillSpread()
        # elif    diffuse_model == "sutton"   : self.spread = SuttonSpread()
        else: raise ValueError("Unknown model")

        if stab_class is not None:
            self.stab_class = stab_class
            self.weather = inverse_stab_class_to_wether(windspeed, stab_class)
        elif weather is not None:
            self.weather = weather
            self.stab_class = classify_atomosphere_stability(windspeed, weather)

        else: raise ValueError("Need weather or stab_class")

    def update(
            self, 
            *, 
            windspeed=None, 
            wind_direction_deg=None, 
            weather=None, 
            stab_class=None
        ):
        if windspeed is not None:
            self.windspeed = windspeed
        if wind_direction_deg is not None:
            self.wind_direction = np.deg2rad(wind_direction_deg)
        if stab_class is not None:
            self.stab_class = stab_class
            self.weather = inverse_stab_class_to_wether(self.windspeed, stab_class)
        elif weather is not None:
            self.weather = weather
            self.stab_class = classify_atomosphere_stability(self.windspeed, weather)

class Source:
    """
    Represents a collection of point sources.
    Stores [Q, x, y, He] in a 4xN ndarray.
    """
    def __init__(self, Q=None, x=None, y=None, He=None):
        self.profile = np.zeros((0,4), dtype=float)
        if (
            Q is not None and 
            x is not None and 
            y is not None and 
            He is not None
        ):
            self.add(Q,x,y,He)

    def add(self, Q, x, y, He):
        # 配列化
        Q = np.atleast_1d(Q)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        He = np.atleast_1d(He)
        
        assert len(Q) == len(x) == len(y) == len(He), "All inputs must have the same length"
        
        new = np.column_stack([Q, x, y, He])
        self.profile = np.vstack([self.profile, new])


    def clear(self):
        self.profile=np.zeros((0, 4), dtype=float)

class DiffusePlume:
    def __init__(
        self, 
        field:Field, 
        source:Source
    ):
        self.field = field
        self.source = source

    def Concentration(self, x, y, z, *, time=None):

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        x, y, z = np.broadcast_arrays(x, y, z)
        C_total = np.zeros_like(x)

        for q, x_src, y_src, He in tqdm(
            self.source.profile,
            total=self.source.profile.shape[0],
            desc="Intergrate all fauntains",
            bar_format="[{desc}, Remaining {remaining}] {percentage:3.1f}% ({elapsed}) |{bar:20}| [{n}/{total}, {rate_fmt}]",
        ):
            # --- Compute concentration ---
            C_total += plume_kernel(
                x=x-x_src, 
                y=y-y_src, 
                z=z, 
                q=q, 
                He=He,
                model=self.field.spread,
                stab_class=self.field.stab_class,
                windspeed=self.field.windspeed,
                time=time
            )

            # --- Sum contributions from all sources ---
        return C_total


def lidar_to_plume(x, y, z, origin, azim):
    """
    ライダー座標系 -> 風下座標系
    x, y, z : ライダー座標系の座標 (array-like)
    origin : 風下座標系原点 in lidar coordinates [x0, y0, z0]
    theta : ライダーx軸から風下x'軸へのccw回転角 (rad)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    x0, y0, z0 = origin

    # 平行移動
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # 回転
    x_p = np.cos(azim) * dx + np.sin(azim) * dy
    y_p = -np.sin(azim) * dx + np.cos(azim) * dy
    z_p = dz  # zはそのまま
    return x_p, y_p, z_p


class DiffusePlumeLidar():
    """
    ライダー座標系で濃度計算を行うための上位互換クラス。
    内部で風向角に基づきモデル座標へ変換して DiffusePlume に委譲する。
    """

    def __init__(
        self, 
        field:Field,
        wind_direction_deg:float, 
        source:Source,
    ):
        self.field = field
        self.source = source
        self.wind_direction = np.deg2rad(wind_direction_deg)


    def Concentration(self, x, y, z, *, time=None):

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        x, y, z = np.broadcast_arrays(x, y, z)
        C_total = np.zeros_like(x)

        for q, x_src, y_src, He in tqdm(
            self.source.profile,
            total=self.source.profile.shape[0],
            desc="Intergrate all fauntains",
            bar_format="[{desc}, Remaining {remaining}] {percentage:3.1f}% ({elapsed}) |{bar:20}| [{n}/{total}, {rate_fmt}]",
        ):

            # --- Compute concentration ---
            x_p, y_p, z_p = lidar_to_plume(x, y, z, [x_src, y_src, 0], self.wind_direction)
            C_total += plume_kernel(
                x=x_p, 
                y=y_p, 
                z=z_p, 
                q=q, 
                He=He,
                model=self.field.spread,
                stab_class=self.field.stab_class,
                windspeed=self.field.windspeed,
                time=time
            )

            # --- Sum contributions from all sources ---
        return C_total
