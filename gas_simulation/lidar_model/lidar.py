from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import constants as consts
from matplotlib import pyplot as plt

from .optics import overlap
@dataclass
class Coord:
    distance: np.ndarray
    theta_deg: float
    x0: float
    z0: float

    def __post_init__(self):
        theta = np.deg2rad(self.theta_deg)
        self.x = self.x0 + self.distance * np.cos(theta)
        self.z = self.z0 + self.distance * np.sin(theta)
    
    def __getitem__(self, key):
        return self.distance[key], self.x[key], self.z[key]
    
    def get_xz(self, r):
        theta = np.deg2rad(self.theta_deg)
        x = self.x0 + r * np.cos(theta)
        z = self.z0 + r * np.sin(theta)
        return x, z
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
        self.coord = Coord(
            distance=np.arange(dR, end, dR),
            theta_deg=elevation_deg,
            x0=0,
            z0=alt_offset,
        )
        self.dR = dR
        self.E0 = E0
        self.A = A
        self.eta = eta
        self.M = M
        self.q = q

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
    def __init__(self, 
                 lidar:Lidar, 
                 distance:np.ndarray,
                 dR,
                 *, 
                 Bj:float=0.0, 
                 F:float=1.0, 
                 D:float=0.0):
        self.coord = Coord(
            distance=distance,
            theta_deg=lidar.coord.theta_deg,
            x0=lidar.coord.x0,
            z0=lidar.coord.z0,
        )
        self.dR = dR
        self.Bj = Bj
        self.F = F
        self.D = D
        
    def calc(
        self, 
        p_on_R1, 
        p_on_R2, 
        p_off_R1, 
        p_off_R2, 
        dR, 
        d_xs
    ):
        res = np.log(
            (p_on_R1 / p_off_R1) * 
            (p_off_R2 / p_on_R2)
        )
        res /= (dR * d_xs)
        return res

    def stat_error(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):
        f = lambda x:(self.D+(x+self.Bj)*self.F)/(x**2)
        res = f(p_on_R1)+f(p_on_R2)+f(p_off_R1)+f(p_off_R2)
        res = np.sqrt(res)
        res /= 2*dR*d_xs
        return res
        