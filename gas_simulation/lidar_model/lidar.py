from numpy.lib.stride_tricks import sliding_window_view as np_SWV
from scipy import constants as consts

from .optics import overlap
from .utils import Coord


class LidarCalc:
    def __init__(
        self,
        *,
        dR: float = 5.0,
        E0: float = 10.0 * 1e-3, # [J]
        A: float = 0.3,
        M: float = 100 * 60 * 10.0,  # 100 Hz / 1 hour
        eta: float = 0.3,
        q: float = 0.3,
    ):
        self.dR = dR
        self.E0 = E0
        self.A = A
        self.eta = eta
        self.M = M
        self.q = q
    def show_params(self):
        print(f"LidarCalc parameters:")
        print(f" dR : {self.dR :<10.2f} [m]")
        print(f" E0 : {self.E0 :<10.3g} [J]")
        print(f" A  : {self.A  :<10.2g} [m^2]")
        print(f" eta: {self.eta:<10.2f}")
        print(f" M  : {self.M  :<10.3g} [shots]")
        print(f" q  : {self.q  :<10.2f}\n")

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


def signal_swm(coord:Coord, power, window_N:int, dist_axis:int):
    new_power = np_SWV(power, window_shape=window_N, axis=dist_axis).sum(axis=-1)
    new_dist = np_SWV(coord.distance, window_shape=window_N).mean(axis=-1)
    new_coord = coord.with_(distance=new_dist)
    return new_power, new_coord
