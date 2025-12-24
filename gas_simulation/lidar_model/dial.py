import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, Optional
from numpy.lib.stride_tricks import sliding_window_view as np_SWV

from .utils import Coord
from gas_simulation.model import Environment
from gas_simulation.atom import alphas_mol, alphas_aer
from gas_simulation.lidar_model import lidar

Numeric: TypeAlias = float | np.ndarray


@dataclass
class RamanShiftObject:
    wl: Numeric
    xs: Numeric
    p: Numeric


@dataclass
class DialInput:
    obj_s1: RamanShiftObject
    obj_s2: RamanShiftObject
    lidar_coord: Coord
    n_gas: dict[str, Numeric]
    aer_absorp_feat: float
    sumN: int = field(default=1)
    diffN: int = field(default=1)


@dataclass
class DialResult:
    coord: Coord
    wl_on: Numeric
    wl_off: Numeric
    n_true: dict[str, Numeric]
    res: Numeric
    stat_err: Numeric
    cf: Numeric


@dataclass
class DialDebugData:
    mask: bool
    p_on: np.ndarray
    p_off: np.ndarray

    # # 任意：後から増えても壊れない余白
    # meta: Optional[dict] = None


class DialCalc:
    def __init__(
        self,
        Bj: float = 0.0,
        F: float = 1.0,
        D: float = 0.0,
    ):
        self.Bj = Bj
        self.F = F
        self.D = D
        self.f = lambda x: (self.D + self.F * (x + self.Bj)) / (x * x)

    def calc(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):
        res = np.log((p_on_R1 / p_off_R1) * (p_off_R2 / p_on_R2))
        res /= dR * d_xs
        return res

    def stat_error(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):

        res = self.f(p_on_R1) + self.f(p_on_R2) + self.f(p_off_R1) + self.f(p_off_R2)
        res = np.sqrt(res) / (2 * dR * d_xs)
        return res

    def calc_correction_factor(self, alt, wl_on, wl_off, aer_absorp_feat: float, d_xs):
        alpha_mol_on = alphas_mol(wl_on, alt)
        alpha_mol_off = alphas_mol(wl_off, alt)
        d_alpha_mol = alpha_mol_on - alpha_mol_off

        alpha_aer_on = alphas_aer(wl_on, alt, aer_absorp_feat)
        alpha_aer_off = alphas_aer(wl_off, alt, aer_absorp_feat)
        d_alpha_aer = alpha_aer_on - alpha_aer_off

        # d_alpha_gas = {}
        # for key, n in n_gas_est.items():
        #     alpha_gas_on = n*env.gas_profile[key].cross_section(wl_on)
        #     alpha_gas_off = n*env.gas_profile[key].cross_section(wl_off)
        #     d_alpha_gas[key] = alpha_gas_on - alpha_gas_off

        return (d_alpha_mol + d_alpha_aer) / d_xs

    def estimate(self, input: DialInput):
        wl_on, wl_off, d_xs, p_on, p_off, mask = onoff_swapper(
            input.obj_s1, input.obj_s2
        )
        if input.sumN > 1:
            print(f"signal sumuaited {input.sumN} points")
            p_on, input.lidar_coord = lidar.signal_swm(
                input.lidar_coord, p_on, input.sumN, dist_axis=0
            )
            p_off, _ = lidar.signal_swm(
                input.lidar_coord, p_off, input.sumN, dist_axis=0
            )
            input.n_gas = {
                k: np_SWV(v, window_shape=input.sumN).mean(axis=-1)
                for k, v in input.n_gas.items()
            }
        p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, dial_coord, n_true = prepare_diff(
            p_on, p_off, input.lidar_coord, input.n_gas, input.diffN
        )

        res = self.calc(
            p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR[:, None], d_xs[None, :]
        )

        stat_err = self.stat_error(
            p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR[:, None], d_xs[None, :]
        )

        cf = self.calc_correction_factor(
            dial_coord.z[:, None],
            wl_on[None, :],
            wl_off[None, :],
            input.aer_absorp_feat,
            d_xs,
        )
        return (
            DialResult(
                coord=dial_coord,
                wl_on=wl_on,
                wl_off=wl_off,
                n_true=n_true,
                res=res,
                stat_err=stat_err,
                cf=cf,
            ),
            DialDebugData(mask=mask, p_on=p_on, p_off=p_off),
        )


def onoff_swapper(obj_s1: RamanShiftObject, obj_s2: RamanShiftObject):
    mask = obj_s1.xs > obj_s2.xs
    mask2 = mask[np.newaxis, :]

    xs_on = np.where(mask, obj_s1.xs, obj_s2.xs)
    xs_off = np.where(mask, obj_s2.xs, obj_s1.xs)
    d_xs = xs_on - xs_off

    wl_on = np.where(mask, obj_s1.wl, obj_s2.wl)
    wl_off = np.where(mask, obj_s2.wl, obj_s1.wl)

    p_on = np.where(mask2, obj_s1.p, obj_s2.p)
    p_off = np.where(mask2, obj_s2.p, obj_s1.p)

    return wl_on, wl_off, d_xs, p_on, p_off, mask


def prepare_diff(
    p_on: Numeric,
    p_off: Numeric,
    coord: Coord,
    n_gas: dict[str, Numeric],
    diffN: int = 1,
):
    p_on_R1 = p_on[:-diffN]
    p_on_R2 = p_on[diffN:]
    p_off_R1 = p_off[:-diffN]
    p_off_R2 = p_off[diffN:]
    dial_dR = coord.distance[diffN:] - coord.distance[:-diffN]
    dial_coord = coord.with_(
        distance=(coord.distance[diffN:] + coord.distance[:-diffN]) / 2,
    )
    n_true = {k: (v[diffN:] + v[:-diffN]) / 2 for k, v in n_gas.items()}
    return p_on_R1, p_on_R2, p_off_R1, p_off_R2, dial_dR, dial_coord, n_true
