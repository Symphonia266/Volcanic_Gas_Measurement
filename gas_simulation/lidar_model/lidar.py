from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import constants as consts
from matplotlib import pyplot as plt

from .optics import overlap
from gas_simulation import utils
from gas_simulation.atom import alphas_mol, alphas_aer
from gas_simulation.atom import betas_N2, betas_O2

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
        dR: float = 5.0,
        E0: float = 0.1,
        A: float = 0.3,
        M: float = 100 * 60 * 30.0,  # 100 Hz / 1 hour
        eta: float = 0.3,
        q: float = 0.3,
    ):
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
    def __init__(
        self,
        Bj: float = 0.0,
        F: float = 1.0,
        D: float = 0.0,
    ):
        self.Bj = Bj
        self.F = F
        self.D = D

    def calc(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):
        res = np.log((p_on_R1 / p_off_R1) * (p_off_R2 / p_on_R2))
        res /= dR * d_xs
        return res

    def stat_error(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):
        f = lambda x: (self.D + self.F*(x + self.Bj)) / (x*x)
        res = f(p_on_R1) + f(p_on_R2) + f(p_off_R1) + f(p_off_R2)
        res = np.sqrt(res)/(2 * dR * d_xs)
        return res
    
def calc_dial_correction_factor(env, alt, wl_on, wl_off, d_xs):
    alpha_mol_on = alphas_mol(wl_on, alt)
    alpha_mol_off = alphas_mol(wl_off, alt)
    d_alpha_mol = alpha_mol_on - alpha_mol_off

    alpha_aer_on = alphas_aer(wl_on, alt, env.aer_absorp_feat)
    alpha_aer_off = alphas_aer(wl_off, alt, env.aer_absorp_feat)
    d_alpha_aer = alpha_aer_on - alpha_aer_off

    # d_alpha_gas = {}
    # for key, n in n_gas_est.items():
    #     alpha_gas_on = n*env.gas_profile[key].cross_section(wl_on)
    #     alpha_gas_off = n*env.gas_profile[key].cross_section(wl_off)
    #     d_alpha_gas[key] = alpha_gas_on - alpha_gas_off

    return (d_alpha_mol + d_alpha_aer) / d_xs
