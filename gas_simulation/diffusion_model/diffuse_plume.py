# coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from .pasquill_stable_classfication import (
    classify_atomosphere_stability,
    inverse_stab_class_to_wether,
)
from .pasquill_model.pasquill_gifford_spreadwidth import SpreadWidth as PasquillSpread
from .sutton_model.sutton_spreadwidth import SpreadWidth as SuttonSpread
from .func import correct_time


class DiffusePlume:
    def __init__(self, windspeed, *, wether=None, stab_class=None, model="pasquill"):
        if model == "pasquill":
            self.model = PasquillSpread()
        if model == "sutton":
            pass
            # self.model = SuttonSpread(windspeed, wether, stab_class)

        self.windspeed = windspeed
        if stab_class != None:
            self.wether = inverse_stab_class_to_wether(windspeed, stab_class)
            self.stab_class = stab_class
        elif wether != None:
            self.wether = wether
            self.stab_class = classify_atomosphere_stability(windspeed, wether)
        else:
            raise ValueError("ERROR : wether or stability class are empty")

        self.sources = pd.DataFrame(columns=["Q", "x", "y", "z"], dtype=float)

    def update_parameters(
        self, *, windspeed=None, wether=None, stab_class=None, model=None, 
    ):
        if model is not None:
            if model == "pasquill":
                self.model = PasquillSpread()
            if model == "sutton":
                pass
            # self.model = SuttonSpread(windspeed, wether, stab_class)

        if windspeed is not None:
            self.windspeed = windspeed
        if stab_class is not None:
            self.wether = inverse_stab_class_to_wether(windspeed, stab_class)
            self.stab_class = stab_class
        elif wether is not None:
            self.wether = wether
            self.stab_class = classify_atomosphere_stability(windspeed, wether)

    def entry_source(self, Q, x, y, He):
        Q = np.atleast_1d(Q)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        He = np.atleast_1d(He)

        # 各配列の長さを確認
        lengths = np.array([Q.size, x.size, y.size, He.size])
        max_len = np.max(lengths)
        min_len = np.min(lengths)

        def expand(v):
            return np.full(max_len, v[0]) if len(v) == 1 else v

        # 長さが不一致の場合（ブロードキャスト可能か確認）
        if max_len != min_len:
            # ブロードキャストできる条件：スカラー（長さ1）が混ざっている場合のみ
            if not np.all((lengths == 1) | (lengths == max_len)):
                raise ValueError(
                    f"Inconsistent array lengths: Q={len(Q)}, x={len(x)}, y={len(y)}, He={len(He)}"
                )
            # スカラーをブロードキャスト（長さmax_lenに拡張）
            else:
                Q, x, y, He = map(expand, (Q, x, y, He))

        source = pd.DataFrame(
            {
                "Q": np.atleast_1d(Q),  # 放出強度
                "x": np.atleast_1d(x),  # 煙源X座標
                "y": np.atleast_1d(y),  # 煙源Y座標
                "He": np.atleast_1d(He),  # 有効煙源高度
            },
            dtype=float,
        )
        self.sources = pd.concat([self.sources, source], ignore_index=True)

    def clear_source(self):
        self.sources = pd.DataFrame(columns=["Q", "x", "y", "He"], dtype=float)

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
            if x.ndim == 1:
                x = x[:, np.newaxis, np.newaxis]
            if y.ndim == 1:
                y = y[np.newaxis, :, np.newaxis]
            if z.ndim == 1:
                z = z[np.newaxis, np.newaxis, :]

        if time_correction is not None:
            windspeed = self.windspeed * time_correction
        else:
            windspeed = self.windspeed * 3 * 60

        for i, source in tqdm(
            self.sources.iterrows(),
            total=self.sources.shape[0],
            desc="Intergrate all fauntains",
            bar_format="[{desc}, Remaining {remaining}] {percentage:3.1f}% ({elapsed}) |{bar:20}| [{n}/{total}, {rate_fmt}]",
        ):
            x = x - source["x"]
            y = y - source["y"]

            # mask points downwind of the source (x relative to source > 0)
            mask = x > 0
            # compute spread widths only at masked (downwind) points
            sigma_y = self.model.lateral(x[mask], self.stab_class)
            sigma_z = self.model.vertical(x[mask], self.stab_class)

            if time_correction:
                Q = source["Q"] * time_correction
                sigma_y = correct_time(time_correction, sigma_y)
                sigma_z = correct_time(time_correction, sigma_z)
            else:
                Q = source["Q"] * 3 * 60

            # build concentration only at masked indices so RHS length matches mask.sum()
            C_i_mask = (
                Q
                / (2 * np.pi * sigma_y * sigma_z * windspeed)
                * np.exp(-(y[mask] ** 2) / (2 * sigma_y**2))
                * (
                    np.exp(-((z[mask] + source["He"]) ** 2) / (2 * sigma_z**2))
                    + np.exp(-((z[mask] - source["He"]) ** 2) / (2 * sigma_z**2))
                )
            )
            C_i[mask] = C_i_mask
            C_i[~mask] = 0
            C_total += C_i
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


class DiffusePlumeLidar(DiffusePlume):
    """
    ライダー座標系で濃度計算を行うための上位互換クラス。
    内部で風向角に基づきモデル座標へ変換して DiffusePlume に委譲する。
    """

    def __init__(
        self, windspeed, wind_direction_deg, *, wether=None, stab_class=None, model="pasquill"
    ):
        super().__init__(windspeed=windspeed, wether=wether, stab_class=stab_class, model=model)
        self.wind_direction = np.deg2rad(wind_direction_deg)
        # self.elevation = np.deg2rad(elevation)

    def update_parameters(
        self,
        *,
        model=None,
        windspeed=None,
        wind_direction=None,
        wether=None,
        stab_class=None,
    ):
        super().update_parameters(
            windspeed=windspeed, wether=wether, stab_class=stab_class, model=model,
        )
        if wind_direction is not None:
            self.wind_direction = np.deg2rad(wind_direction)

        # self.elevation = np.deg2rad(elevation)

    def Concentration(self, x, *, y=0, z=0, time_correction=None):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        # --- broadcast arrays so all shapes match ---
        x, y, z = np.broadcast_arrays(x, y, z)
        C_total = np.zeros_like(x, dtype=float)

        # --- windspeed correction ---
        if time_correction is not None:
            windspeed = self.windspeed * time_correction
        else:
            windspeed = self.windspeed * 3 * 60

        for i, source in tqdm(
            self.sources.iterrows(),
            total=self.sources.shape[0],
            desc="Intergrate all fauntains",
            bar_format="[{desc}, Remaining {remaining}] {percentage:3.1f}% ({elapsed}) |{bar:20}| [{n}/{total}, {rate_fmt}]",
        ):
            # use the broadcasted grid arrays so shapes of x_p, y_p, z_p match
            x_p, y_p, z_p = lidar_to_plume(
                x,
                y,
                z,
                [source["x"], source["y"], 0],
                self.wind_direction,
            )
            mask = x_p > 0

            # compute spread widths only at downwind points
            sigma_y = np.zeros_like(x_p)
            sigma_z = np.zeros_like(x_p)
            sigma_y[mask] = self.model.lateral(x_p[mask], self.stab_class)
            sigma_z[mask] = self.model.vertical(x_p[mask], self.stab_class)

            if time_correction:
                Q = source["Q"] * time_correction
                sigma_y[mask] = correct_time(time_correction, sigma_y[mask])
                sigma_z[mask] = correct_time(time_correction, sigma_z[mask])
            else:
                Q = source["Q"] * 3 * 60

            C = np.zeros_like(x_p, dtype=float)
            C[mask] = (
                Q
                / (2 * np.pi * sigma_y[mask] * sigma_z[mask] * windspeed)
                * np.exp(-(y_p[mask] ** 2) / (2 * sigma_y[mask] ** 2))
                * (
                    np.exp(-((z_p[mask] - source["He"]) ** 2) / (2 * sigma_z[mask] ** 2))
                    + np.exp(
                        -((z_p[mask] + source["He"]) ** 2) / (2 * sigma_z[mask] ** 2)
                    )
                )
            )
            C_total += C

        return C_total
