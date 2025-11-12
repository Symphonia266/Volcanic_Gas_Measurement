import numpy as np
import re
from functools import cache
from .. import package_path, cmap


@cache
def N(z):
    """atmosphere number density of molecules [m-3]"""
    return 2.60e25 * np.exp(-1.08e-4 * z)


def Aero(wl, z):
    """Aerozol model"""
    return betas_mol(wl, z) * 36.31 * np.exp(-8.78e-4 * z) * np.power(wl / 710, 4)


def xses_ray(wl):
    """:math:`\\sigma` Cross-sectional area of absorption for Rayleigh scattering [m**2]"""
    return 5.45e-32 * (550 / wl) ** 4


def betas_mol(wl, z):
    """the backscatter coefficient of atmospheric moleculars [m-1]"""
    return xses_ray(wl) * N(z)


def betas_aer(wl, feat, z):
    """the backscatter coefficient of aerozols [m-1]"""
    # if wl > 400:
    #     return Aero(wl, z) * np.power((710 / wl), feat)
    # else:
    #     return Aero(wl, z) * np.power((710 / wl), feat)
    return Aero(wl, z) * np.power((710 / wl), feat)


def alphas_mol(wl, z):
    """the dissipation coefficient of atmospheric moleculars [m-1]"""
    return 8 * np.pi * betas_mol(wl, z) / 3


def alphas_aer(wl, feat, z):
    """the dissipation coefficient of aerozol [m-1]"""
    return 30 * betas_aer(wl, feat, z)

class AtomModel:
    def __init__(self, ):
        self.wl = wl
        self.feat = feat

    def beta(self, z):
        return betas_mol(self.wl, z) + betas_aer(self.wl, self.feat, z)

    def alpha(self, z):
        return alphas_mol(self.wl, z) + alphas_aer(self.wl, self.feat, z)