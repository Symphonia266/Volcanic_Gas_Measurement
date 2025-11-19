from turtle import distance
import numpy as np
import pandas as pd

from scipy import constants as consts
from .consts import gases, s_raman_N2, s_raman_O2

from matplotlib import pyplot as plt
from .atom import (
    N, 
    alphas_aer, 
    alphas_mol, 
    betas_aer, 
    betas_mol
)
from . import utils
from .optics import overlap

class Lidar :
    def __init__(
        self, 
        *,
        end = 1000,
        dR: float = 5,
        elevation=0,
        alt_offset=1,

        E0:float=0.1, 
        A:float=0.3, 
        M: float = 100 * 60 * 60,  # 100 Hz / 1 hour
        eta: float = 0.3,
        q: float = 0.3,
    ):  
        self.distance = np.arange(0, end, dR)
        self.dR = dR 
        self.elevation = elevation
        self.x_grid = self.distance * np.cos(np.deg2rad(elevation))
        self.z_grid = self.distance * np.sin(np.deg2rad(elevation))+alt_offset

        self.E0 = E0
        self.A = A
        self.eta = eta
        self.M = M
        self.q = q
        # self.Bj = Bj
        # self.F = F
        # self.D = D

    def power(self, wl, beta, tau):
        dist = np.atleast_1d(self.distance)[:, np.newaxis]
        wl = np.atleast_1d(wl)[np.newaxis, :]

        t1 = (
            self.E0
            *self.dR
            *self.A
            *self.eta
            *self.M
            *self.q
            /consts.h
            *overlap(dist)
            /dist**2
            *wl
            *1e-9
            /consts.c
        )
        return t1*beta*tau
        
class DIAL:
    def __init__(
        self,         
        distance,
        wl_on, 
        wl_off,
        Bj: float = 0,
        F: float = 1,
        D: float = 0,
    ):
      sys = Lidar()
      p_on = sys.power(wl_on)
      p_on = sys.power(distance, wl_on)
