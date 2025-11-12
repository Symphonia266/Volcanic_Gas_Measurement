# coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from .pasquill_stable_classfication import (
    classification_table,
    classification_label,
    classify_atomosphere_stability,
    inverse_stab_class_to_wether
)
from .pasquill_model.pasquill_gifford_spreadwidth import SpreadWidth as PasquillSpread
from .sutton_model.sutton_spreadwidth import SpreadWidth as SuttonSpread
from .func import correct_time

class DiffusePlume():
    def __init__(self, model, windspeed, *,  wether, stab_class):
        self.windspeed = windspeed
        if stab_class != None:
            self.wehter = inverse_stab_class_to_wether(windspeed, stab_class)
            self.stab_class = stab_class
        elif wether != None:
            self.wether = wether
            self.stab_class = classify_atomosphere_stability(windspeed, wether)
        else:
             raise ValueError("ERROR : wether or stability class are empty")
        
        if model == "pasquill":
            self.model = PasquillSpread()
        if model == "sutton":
            pass
            # self.model = SuttonSpread(windspeed, wether, stab_class)
        self.sources = pd.DataFrame(columns=["Q", "x", "y", "z"], dtype=float)

    def entry_source(self, Q, x, y, z):
        Q = np.atleast_1d(Q)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        # 各配列の長さを確認
        lengths = np.array([Q.size, x.size, y.size, z.size])
        max_len = np.max(lengths)
        min_len = np.min(lengths)
        def expand(v):
            return np.full(max_len, v[0]) if len(v) == 1 else v
                
        # 長さが不一致の場合（ブロードキャスト可能か確認）
        if max_len != min_len:
            # ブロードキャストできる条件：スカラー（長さ1）が混ざっている場合のみ
            if not np.all( (lengths == 1) | (lengths == max_len) ):
                raise ValueError(
                    f"Inconsistent array lengths: Q={len(Q)}, x={len(x)}, y={len(y)}, z={len(z)}"
                )
            # スカラーをブロードキャスト（長さmax_lenに拡張）
            else: Q, x, y, z = map(expand, (Q, x, y, z))

        source = pd.DataFrame({
            "Q": np.atleast_1d(Q),  # 放出強度
            "x": np.atleast_1d(x),  # 煙源X座標
            "y": np.atleast_1d(y),  # 煙源Y座標
            "z": np.atleast_1d(z),  # 煙源Z座標
        }, dtype=float)
        self.sources = pd.concat([self.sources, source], ignore_index=True)

    def clear_source(self):
        self.sources = pd.DataFrame(columns=["Q", "x", "y", "z"], dtype=float)

    def Concentration(self, x_in, y_in, z_in, *, time_correction=None):
        x = np.asarray(x_in)
        y = np.asarray(y_in)
        z = np.asarray(z_in)

        # if "x", "y", and "z" are not gridpoint
        if (x.shape == y.shape) and (y.shape == z.shape): 
            C_i = np.empty_like(x, dtype=float)
            C_total = np.zeros_like(x, dtype=float)
        else:
            C_i = np.empty((x.size, y.size, z.size), dtype=float)
            C_total = np.zeros((x.size, y.size, z.size), dtype=float)
            # reshape them for broadcast
            if x.ndim == 1: x = x[:, np.newaxis, np.newaxis]
            if y.ndim == 1: y = y[np.newaxis, :, np.newaxis]
            if z.ndim == 1: z = z[np.newaxis, np.newaxis, :]
        

        if time_correction is not None:
            windspeed = self.windspeed * time_correction
        else: 
            windspeed = self.windspeed *3*60
        
        for i, source in tqdm(
            self.sources.iterrows(), 
            total=self.sources.shape[0],
            desc="Intergrate all fauntains",
            bar_format="[{desc}, Remaining {remaining}] {percentage:3.1f}% ({elapsed}) |{bar:20}| [{n}/{total}, {rate_fmt}]"
        ):
            mask = (x-source["x"])>0
            sigma_y = self.model.lateral(x[mask]-source["x"], self.stab_class)
            sigma_z = self.model.vertical(x[mask]-source["x"], self.stab_class)

            if time_correction:
                Q = source["Q"] * time_correction
                sigma_y = correct_time(time_correction, sigma_y)
                sigma_z = correct_time(time_correction, sigma_z)
            else:
                Q = source["Q"] * 3*60

            C_i[mask] = (
                Q / (2 * np.pi * sigma_y * sigma_z * windspeed)
                * np.exp( -((y[mask]-source["y"])**2) / (2 * sigma_y**2) )
                *(  
                    np.exp(-((z[mask] + source["z"]) ** 2) / (2 * sigma_z**2))
                    + np.exp(-((z[mask] - source["z"]) ** 2) / (2 * sigma_z**2))
                ) 
            )
            C_i[~mask] = 0
            C_total += C_i
        return C_total

def rotate_lidar_to_plume(x, y, z, theta, phi):
    """
    x, y, z : ライダー座標系
    theta   : 風向方位角 (rad, CCW+)
    phi     : ライダー視線仰角 (rad, 上向き+)
    """

    # --- yaw (θ) ---
    x1 =  x*np.cos(theta) + y*np.sin(theta)
    y1 = -x*np.sin(theta) + y*np.cos(theta)
    z1 = z

    # --- pitch (φ) ---
    xw = x1*np.cos(phi) - z1*np.sin(phi)
    yw = y1
    zw = x1*np.sin(phi) + z1*np.cos(phi)

    return xw, yw, zw


class DiffusePlumeLidar():
    """
    ライダー座標系で濃度計算を行うための上位互換クラス。
    内部で風向角に基づきモデル座標へ変換して DiffusePlume に委譲する。
    """
    def __init__(self, model, windspeed, wind_direction_deg, elevation, *, wether=None, stab_class=None):
        self.core = DiffusePlume(model=model, windspeed=windspeed, wether=wether, stab_class=stab_class)
        self.wind_direction = np.deg2rad(wind_direction_deg)
        self.elevation = np.deg2rad(elevation)

    def update_azim_elev(self, wind_direction_deg, elevation):
        self.wind_direction = np.deg2rad(wind_direction_deg)
        self.elevation = np.deg2rad(elevation)

    def entry_source(self, Q, x_lidar, y_lidar, z_lidar):
        x_wind, y_wind, z_wind = rotate_lidar_to_plume(
            x_lidar, 
            y_lidar, 
            z_lidar, 
            self.wind_direction, 
            self.elevation
        )
        self.core.entry_source(Q, x_wind, y_wind, z_wind)

    def Concentration(self, x_lidar, *, y_lidar=0.0, z_lidar=0.0, time_correction=None):
        # 計算点もライダー座標 → 風下座標へ
        x_wind, y_wind, z_wind = rotate_lidar_to_plume(
            np.asarray(x_lidar), 
            np.asarray(y_lidar), 
            np.asarray(z_lidar),
            self.wind_direction, 
            self.elevation
        )
        return self.core.Concentration(x_wind, y_wind, z_wind, time_correction=time_correction)
