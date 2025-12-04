import numpy as np
import pandas as pd
from scipy import constants as consts
from matplotlib import pyplot as plt

from .optics import overlap

def rotate(r, theta_deg, origin):
    """
    ライダー座標系 -> 風下座標系
    x, y, z : ライダー座標系の座標 (array-like)
    origin : 風下座標系原点 in lidar coordinates [x0, y0, z0]
    theta : ライダーx軸から風下x'軸へのccw回転角 (rad)
    """
    a0, b0 = origin
    a = a0 + r * np.cos(np.deg2rad(theta_deg))
    b = b0 + r * np.sin(np.deg2rad(theta_deg))
    return a, b

class Lidar:
    def __init__(
        self,
        *,
        end: float = 100.0,
        dR: float = 5.0,
        elevation_deg=0.0,
        alt_offset=1.0,
        E0: float = 0.1,
        A: float = 0.3,
        M: float = 100 * 60 * 60.0,  # 100 Hz / 1 hour
        eta: float = 0.3,
        q: float = 0.3,
    ):
        self.distance = np.arange(0, end, dR)
        self.dR = dR
        self.elevation_deg = elevation_deg
        self.alt_offset = alt_offset

        self.x_grid, self.z_grid = rotate(
            self.distance, self.elevation_deg, [0, alt_offset]
        )
        self.E0 = E0
        self.A = A
        self.eta = eta
        self.M = M
        self.q = q
        # self.Bj = Bj
        # self.F = F
        # self.D = D

    def power(self, dist, wl, beta_tau):
        t1 = (
            self.E0
            * self.dR
            * self.A
            * self.eta
            * self.M
            * self.q
            / consts.h
            * overlap(dist)
            / dist**2
            * wl
            * 1e-9
            / consts.c
        )

        return t1 * beta_tau

class Dial:
    def __init__(self, lidar:Lidar, distance:np.ndarray):
        self.distance = distance
        self.dR = np.diff(self.distance)
        self.x_grid, self.z_grid = rotate(
            self.distance, 
            lidar.elevation_deg, 
            [0, lidar.alt_offset]
        )
        
    def calc(
        self, 
        p_on_front, 
        p_on_back, 
        p_off_front, 
        p_off_back, 
        dR, 
        d_xs
    ):
        res = np.log(
            (p_on_front/p_on_back) * 
            (p_off_back/p_off_front)
        )
        res /= (dR * d_xs)
        return res
