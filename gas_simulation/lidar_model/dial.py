from ctypes import util
from math import tau
from tkinter import N
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Protocol, TypeAlias, Self
from numpy.lib.stride_tricks import sliding_window_view as np_SWV

from .utils import Coord
from gas_simulation import utils
from gas_simulation.model import Environment
from gas_simulation.atom import alphas_mol, alphas_aer
from gas_simulation.lidar_model import lidar

Numeric: TypeAlias = float | np.ndarray
_UNSET = object()


@dataclass
class RamanShiftObject:
    wl: np.ndarray
    xs: np.ndarray
    p: np.ndarray


@dataclass
class DialInput:
    env: Environment
    obj_s1: RamanShiftObject
    obj_s2: RamanShiftObject
    lidar_coord: Coord
    env: Environment
    sumN: int = field(default=1)

    def __post_init__(self):
        self.n_gas: dict[str, np.ndarray] = self.env.number_density_at(
            self.lidar_coord.x, 0, self.lidar_coord.z
        )

    def with_(
        self,
        *,
        obj_s1: RamanShiftObject | object = _UNSET,
        obj_s2: RamanShiftObject | object = _UNSET,
        lidar_coord: Coord | object = _UNSET,
        env: Environment | object = _UNSET,
        sumN: int | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            obj_s1=self.obj_s1 if obj_s1 is _UNSET else obj_s1,
            obj_s2=self.obj_s2 if obj_s2 is _UNSET else obj_s2,
            lidar_coord=self.lidar_coord if lidar_coord is _UNSET else lidar_coord,
            env=self.env if env is _UNSET else env,
            sumN=self.sumN if sumN is _UNSET else sumN,
        )


@dataclass
class DialResult:
    coord: Coord
    wl_on: np.ndarray
    wl_off: np.ndarray
    n_true: dict[str, np.ndarray]
    res: np.ndarray
    stat_err: np.ndarray

@dataclass
class DialDebugData:
    coord: Coord
    mask: bool
    p_on: np.ndarray
    p_off: np.ndarray
    d_xs: np.ndarray

@dataclass
class CorrFactorInput:
    alt: np.ndarray
    wl_on: np.ndarray
    wl_off: np.ndarray
    env: Environment
    d_xs: np.ndarray

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

    def show_params(self):
        print(f"Dial parameters:")
        print(f" Bj : {self.Bj:<10.3g}")
        print(f" F  : {self.F :<10.3g}")
        print(f" D  : {self.D :<10.3g}\n")

    def calc(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):
        res = np.log((p_on_R1 / p_off_R1) * (p_off_R2 / p_on_R2))
        res /= dR * d_xs
        return res

    def stat_error(self, p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, d_xs):

        res = self.f(p_on_R1) + self.f(p_on_R2) + self.f(p_off_R1) + self.f(p_off_R2)
        res = np.sqrt(res) / (2 * dR * d_xs)
        return res


    def estimate(
        self, 
        input: DialInput, 
    ) -> tuple[DialResult, CorrFactorInput,  DialDebugData]:
        
        wl_on, wl_off, d_xs, p_on, p_off, mask = onoff_swapper(
            input.obj_s1, input.obj_s2
        )
        if input.sumN > 1:
            print(f"signal sumuaited {input.sumN} points")
            p_on, lidar_coord = lidar.signal_swm(
                input.lidar_coord, p_on, input.sumN, dist_axis=0
            )
            p_off, _ = lidar.signal_swm(
                input.lidar_coord, p_off, input.sumN, dist_axis=0
            )
            n_gas = {
                k: np_SWV(v, window_shape=input.sumN).mean(axis=-1)
                for k, v in input.n_gas.items()
            }
        else:
            lidar_coord = input.lidar_coord
            n_gas = input.n_gas

        p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR, dial_coord, n_true = prepare_diff(
            p_on, p_off, lidar_coord, n_gas, input.sumN
        )

        res = self.calc(
            p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR[:, None], d_xs[None, :]
        )

        stat_err = self.stat_error(
            p_on_R1, p_on_R2, p_off_R1, p_off_R2, dR[:, None], d_xs[None, :]
        )

        return (
            DialResult(
                coord=dial_coord,
                wl_on=wl_on,
                wl_off=wl_off,
                n_true=n_true,
                res=res,
                stat_err=stat_err,
            ),
            CorrFactorInput(
                alt=dial_coord.z,
                wl_on=wl_on,
                wl_off=wl_off,
                env=input.env,
                d_xs=d_xs,
            ),
            DialDebugData(
                coord=lidar_coord, 
                mask=mask, 
                p_on=p_on, 
                p_off=p_off, 
                d_xs=d_xs
            ),
        )


def onoff_swapper(obj_s1: RamanShiftObject, obj_s2: RamanShiftObject):
    mask = obj_s1.xs > obj_s2.xs
    xs_on = np.where(mask, obj_s1.xs, obj_s2.xs)
    xs_off = np.where(mask, obj_s2.xs, obj_s1.xs)
    d_xs = xs_on - xs_off

    wl_on = np.where(mask, obj_s1.wl, obj_s2.wl)
    wl_off = np.where(mask, obj_s2.wl, obj_s1.wl)

    mask2 = mask[np.newaxis, :]
    p_on = np.where(mask2, obj_s1.p, obj_s2.p)
    p_off = np.where(mask2, obj_s2.p, obj_s1.p)

    return wl_on, wl_off, d_xs, p_on, p_off, mask


def prepare_diff(
    p_on: np.ndarray,
    p_off: np.ndarray,
    coord: Coord,
    n_gas: dict[str, np.ndarray],
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

def calc_correction_factor(
    input: CorrFactorInput,
    *,
    mol = True,
    aer = True,
    n_gas_est: dict[str, np.ndarray] | None = None,
):
    d_alpha_others = 0.0
    if mol:
        alpha_mol_on = alphas_mol(input.wl_on[None, :], input.alt[:, None])
        alpha_mol_off = alphas_mol(input.wl_off[None, :], input.alt[:, None])    
        d_alpha_others = d_alpha_others + alpha_mol_on - alpha_mol_off
    # print(d_alpha_others[~np.isnan(d_alpha_others)][0])
    if aer:
        alpha_aer_on = alphas_aer(input.wl_on[None, :], input.alt[:, None], input.env.aer_absorp_feat)
        alpha_aer_off = alphas_aer(input.wl_off[None, :], input.alt[:, None], input.env.aer_absorp_feat)
        d_alpha_others = d_alpha_others + alpha_aer_on - alpha_aer_off

    if n_gas_est is not None:
        for key, n in n_gas_est.items():
            # print(f"{key} : {n[0]}")
            n = utils.ppm_to_number_density(n, input.alt)
            xs_on = input.env.gas_inventory[key].cross_section(input.wl_on)
            xs_off = input.env.gas_inventory[key].cross_section(input.wl_off)
            alpha_gas_on = n[:, None] * xs_on[None, :]
            alpha_gas_off = n[:, None] * xs_off[None, :]

            d_alpha_gas = alpha_gas_on - alpha_gas_off
            d_alpha_others = d_alpha_others + d_alpha_gas
    # else:
    #     print("No gas contamination correction applied.")

    return d_alpha_others / input.d_xs
