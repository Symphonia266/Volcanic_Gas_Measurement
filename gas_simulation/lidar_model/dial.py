import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as np_SWV
from .utils import Coord
from gas_simulation.model import Environment
from gas_simulation.atom import betas_N2, betas_O2
from gas_simulation.atom import alphas_mol, alphas_aer


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

    def calc(self, 
        p_on_R1, 
        p_on_R2, 
        p_off_R1, 
        p_off_R2, 
        dR, 
        d_xs
    ):
        res = np.log((p_on_R1 / p_off_R1) * (p_off_R2 / p_on_R2))
        res /= dR * d_xs
        return res

    def stat_error(self, 
        p_on_R1, 
        p_on_R2, 
        p_off_R1, 
        p_off_R2, 
        dR, 
        d_xs
    ):

        res = (
            self.f(p_on_R1) + 
            self.f(p_on_R2) + 
            self.f(p_off_R1) + 
            self.f(p_off_R2)
        )
        res = np.sqrt(res) / (2 * dR * d_xs)
        return res

def onoff_swapper(
    wl_s1, wl_s2, 
    xs_s1, xs_s2, 
    p_s1, p_s2, 
):
    mask = xs_s1 > xs_s2
    mask2 = mask[np.newaxis, :]

    xs_on = np.where(mask, xs_s1, xs_s2)
    xs_off = np.where(mask, xs_s2, xs_s1)
    d_xs = xs_on - xs_off

    wl_on = np.where(mask, wl_s1, wl_s2)
    wl_off = np.where(mask, wl_s2, wl_s1)

    p_on = np.where(mask2, p_s1, p_s2)
    p_off = np.where(mask2, p_s2, p_s1)

    return wl_on, wl_off, d_xs, p_on, p_off, mask

def prepare_diff(p_on, p_off, coord, n_gas, diffN=1):
    diffN = 1
    p_on_R1 = p_on[:-diffN]
    p_on_R2 = p_on[diffN:]
    p_off_R1 = p_off[:-diffN]
    p_off_R2 = p_off[diffN:]
    dial_dR = coord.distance[diffN:] - coord.distance[:-diffN]
    dial_coord = coord.with_(
        distance=(coord.distance[diffN:] + coord.distance[:-diffN]) / 2,
    )
    n_true = {
        k : (v[diffN:]+v[:-diffN])/2 
        for k, v in n_gas.items()
    }
    return p_on_R1, p_on_R2, p_off_R1, p_off_R2, dial_dR, dial_coord, n_true

def calc_dial_correction_factor(
    env: Environment, alt, wl_on, wl_off, d_xs
):
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
