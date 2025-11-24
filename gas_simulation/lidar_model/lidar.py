import numpy as np
import pandas as pd
from scipy import constants as consts
from matplotlib import pyplot as plt

from .optics import overlap


class Lidar:
    def __init__(
        self,
        *,
        end:float=100,
        dR: float = 5,
        elevation=0,
        alt_offset=1,
        E0: float = 0.1,
        A: float = 0.3,
        M: float = 100 * 60 * 60,  # 100 Hz / 1 hour
        eta: float = 0.3,
        q: float = 0.3,
    ):
        self.distance = np.arange(0, end, dR)
        self.dR = dR
        self.elevation = elevation
        self.alt_offset = alt_offset
        
        self.x_grid = self.distance * np.cos(np.deg2rad(elevation))
        self.z_grid = self.distance * np.sin(np.deg2rad(elevation)) + alt_offset

        self.E0 = E0
        self.A = A
        self.eta = eta
        self.M = M
        self.q = q
        # self.Bj = Bj
        # self.F = F
        # self.D = D

    def power(self, wl, beta, tau):
        dist = np.atleast_1d(self.distance)[1:, np.newaxis]
        wl = np.atleast_1d(wl)[np.newaxis, :]

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
        return t1 * beta * tau


class Dial:
    def __init__(self, lidar, dR):
        self.lidar = lidar
        self.dR = dR[:, np.newaxis]
        self.distance = (lidar.distance[2:] + lidar.distance[1:-1]) / 2
        self.x_grid = (lidar.x_grid[2:] + lidar.x_grid[1:-1]) / 2
        self.z_grid = (lidar.z_grid[2:] + lidar.z_grid[1:-1]) / 2

    def concentration(self, p_on, p_off, d_xs):
        res = np.log(
            (p_on[:-1, :]/p_on[1:, :]) * 
            (p_off[1:, :]/p_off[:-1, :])
        )
        res /= (self.dR * d_xs)
        return res
