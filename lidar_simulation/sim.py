
# from matplotlib.layout_engine import ConstrainedLayoutEngine
import copy
import re
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy.interpolate import interp1d
from cycler import cycler

import lidar_simulation.func.atom_model as atm
from .func import *
from .consts import *

# read file and reformatting
xs_SO2_298K_raw = pd.read_excel(
    package_path+f"input/SO2_VandaeleHermansFally(2009)_298K_227.275-416.658nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
)
xs_SO2_358K_raw = pd.read_excel(
    package_path+f"input/SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
)

xs_O3_293K_raw = pd.read_excel(
    package_path+f"input/O3_Bogumil(2003)_293K_230-1070nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
)

xs_H2S_294K_raw = pd.read_excel(
    package_path+f"input/H2S_Grosch(2015)_294.8K_198-370nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
)
xs_H2S_423K_raw = pd.read_excel(
    package_path+f"input/H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
)

class rDIAL_sim_package:
    # === Lidar equation simulation parameter ===

    def __init__(self, scat1: str, way1: bool, scat2: str, way2: bool):
        """
        Args:
            scat1 (_str_): the Raman scatterer("O2" or "N2")
            way1 (_bool_): the direction of Raman shift, ANTI-STOKES is True
            scat2 (_str_): the Raman scatterer("O2" or "N2"), anoter one
            way2 (_bool_): the direction of Raman shift, anoter one, ANTI-STOKES is True
        """
        self.wl_sim_start: float = 230
        self.wl_sim_end: float = 370
        self.wl_step = 0.002
        self.range_sim = [0, 1000]
        self.range_dark_gas = [600, 800]

        self.out_dir = f"/{scat1}-{("AS" if way1 else "ST")}_{scat2}-{("AS" if way2 else "ST")}/eff/"
        self.scat1 = scat1
        self.scat2 = scat2
        self.way1 = way1
        self.way2 = way2
        self.lb1 = f"{self.scat1}_{"AS" if self.way1 else "ST"}"
        self.lb2 = f"{self.scat2}_{"AS" if self.way2 else "ST"}"

        # self.flg: np.ndarray
        # self.p_on: pd.DataFrame
        # self.p_off: pd.DataFrame

        # self.xs_O3: pd.DataFrame
        # """ the CrossSection of O3"""
        # self.xs_H2S: dict[int, pd.DataFrame]
        # """ the CrossSection of H2S, with thin (0, FALSE) or thick (1, TRUE)"""
        # self.xs_SO2: dict[int, pd.DataFrame]
        # """ the CrossSection of SO2, with thin (0, FALSE) or thick (1, TRUE)"""

        # # self.d_xs_SO2: pd.DataFrame

        # self.d_alpha_all: pd.DataFrame
        # self.d_alpha_mol: pd.DataFrame
        # self.d_alpha_aer: pd.DataFrame

        # self.n1: pd.DataFrame
        # self.way2: pd.DataFrame
        # self.n3: pd.DataFrame

        # self.error1: pd.DataFrame
        # self.error2: pd.DataFrame
        # self.error3: pd.DataFrame

        # self.stdev_n: pd.DataFrame
        # self.wl_opt: int | float | str

    def entry_param(
        self,
        E0:float=0.1, 
        A:float=0.3, 
        M: float = 100 * 60 * 60,  # 100 Hz / 1 hour
        eta: float = 0.3,
        q: float = 0.3,
        dR: float = 30,
        Bj: float = 0,
        F: float = 1,
        D: float = 0,
        alt: float = 1000
    ):
        """
        Args:
            E0  (_float_): the average energy of palse [J]
            A   (_float_): the diameter of the receiving telescope [m]
            M   (_float_): the number of times of integration
            eta (_float_): the quantum efficiency of the detector
            q   (_float_): the total efficiency of optics
            dR  (_float_): the step of simulate [m]
            Bj  (_float_): the background light noise
            F   (_float_): the noise of detectors
            D   (_float_): dark count
        """
        self.E0=E0
        self.A=A
        self.M=M
        self.eta=eta 
        self.q=q
        self.dR=dR
        self.Bj=Bj
        self.F=F
        self.D=D
        self.alt=alt
        self.int_R: np.ndarray = np.arange(
            self.range_sim[0],
            self.range_sim[1],
            self.dR,
        )
        self.R:np.ndarray = self.int_R[1:]
        self.aveR = (self.R[1:] + self.R[:-1]) / 2

        self.wl = np.arange(self.wl_sim_start, self.wl_sim_end, self.wl_step)
        self.wl_scat1 = wl_shift(self.wl, gases.at[self.scat1, "sft"], dir=self.way1)
        self.wl_scat2 = wl_shift(self.wl, gases.at[self.scat2, "sft"], dir=self.way2)

        self.xses_SO2_358K = interp1d(xs_SO2_358K_raw["WL[nm]"], xs_SO2_358K_raw["XS[m2/mol]"], bounds_error=False, fill_value=np.nan)
        self.xses_SO2_298K = interp1d(xs_SO2_298K_raw["WL[nm]"], xs_SO2_298K_raw["XS[m2/mol]"], bounds_error=False, fill_value=np.nan)
        self.xses_H2S_423K = interp1d(xs_H2S_423K_raw["WL[nm]"], xs_H2S_423K_raw["XS[m2/mol]"], bounds_error=False, fill_value=np.nan)
        self.xses_H2S_294K = interp1d(xs_H2S_294K_raw["WL[nm]"], xs_H2S_294K_raw["XS[m2/mol]"], bounds_error=False, fill_value=np.nan)
        self.xses_O3_293K  = interp1d(xs_O3_293K_raw["WL[nm]"],  xs_O3_293K_raw["XS[m2/mol]"] , bounds_error=False, fill_value=np.nan)

        # self.n_SO2 = interp1d(r_obs, SO2_obs)
        # self.n_H2S = interp1d(r_obs, H2S_obs)
        self.n_O3 = N(self.alt) * 0.005 * 1e-6
        """ the density of O3 [ppm] """

    def judge_dark(self, r):
        r = np.asarray(r)
        return (self.range_dark_gas[0] <= r) & (r < self.range_dark_gas[1])

    def dist_SO2(self,r):
        """ the density of SO2 [ppm] """
        r = np.asarray(r)
        result = np.where(
            self.judge_dark(r),
            atm.N(self.alt) * 45e-6,
            N(self.alt) * 0.07e-6
        )
        return result.item() if np.isscalar(r) else result
    
    def dist_H2S(self,r):
        """ the density of H2S [ppm] """
        r = np.asarray(r)
        result = np.where(
            self.judge_dark(r),
            N(self.alt) * 45e-6,
            N(self.alt) * 0.07e-6
        )
        return result.item() if np.isscalar(r) else result
    
    def powers(self, ft_aer, *, new=True):
        if new == True:
            int_R=self.int_R
            alpha_mol_base = calc_of_effective(self.wl, alphas_mol, args=(self.alt))
            alpha_mol_s1 = calc_of_effective(self.wl_scat1, alphas_mol, args=(self.alt))
            alpha_mol_s2 = calc_of_effective(self.wl_scat2, alphas_mol, args=(self.alt))
            
            alpha_aer_base = calc_of_effective(self.wl, alphas_aer, args=(ft_aer, self.alt))
            alpha_aer_s1 = calc_of_effective(self.wl_scat1, alphas_aer, args=(ft_aer, self.alt))
            alpha_aer_s2 = calc_of_effective(self.wl_scat2, alphas_aer, args=(ft_aer, self.alt))
            
            # === the absorption by SO2 === 
            ## laser absorp
            n_SO2 = self.dist_SO2(int_R)
            i = calc_of_effective(self.wl, self.xses_SO2_358K)
            j = calc_of_effective(self.wl, self.xses_SO2_298K)
            xs_SO2_base = np.array([
                i if self.judge_dark(r) else j
                for r in int_R
            ])
            ## scatterar 1 absorp
            i = calc_of_effective(self.wl_scat1, self.xses_SO2_358K)
            j = calc_of_effective(self.wl_scat1, self.xses_SO2_298K)
            xs_SO2_s1 = np.array([
                i if self.judge_dark(r) else j
                for r in int_R
            ])
            ## scatterar 2 absorp
            i = calc_of_effective(self.wl_scat2, self.xses_SO2_358K)
            j = calc_of_effective(self.wl_scat2, self.xses_SO2_298K)
            xs_SO2_s2 = np.array([
                i if self.judge_dark(r) else j
                for r in int_R
            ])
            flg_s1_on = (xs_SO2_s1 > xs_SO2_s2)[1:, :]
            alpha_SO2_s1 = n_SO2[:, np.newaxis]*(xs_SO2_base+xs_SO2_s1)
            alpha_SO2_s2 = n_SO2[:, np.newaxis]*(xs_SO2_base+xs_SO2_s2)

            # === the absorption by H2S === 
            n_H2S = self.dist_H2S(int_R)
            i = calc_of_effective(self.wl, self.xses_H2S_423K)
            j = calc_of_effective(self.wl, self.xses_H2S_294K)
            alpha_H2S_base = np.array([
                i if self.judge_dark(r) else j
                for r in int_R
            ])*n_H2S[:, np.newaxis]

            i = calc_of_effective(self.wl_scat1, self.xses_H2S_423K)
            j = calc_of_effective(self.wl_scat1, self.xses_H2S_294K)
            alpha_H2S_s1 = alpha_H2S_base + np.array([
                i if self.judge_dark(r) else j
                for r in int_R
            ])*n_H2S[:, np.newaxis]
            
            i = calc_of_effective(self.wl_scat2, self.xses_H2S_423K)
            j = calc_of_effective(self.wl_scat2, self.xses_H2S_294K)
            alpha_H2S_s2 = alpha_H2S_base + np.array([
                i if self.judge_dark(r) else j
                for r in int_R
            ])*n_H2S[:, np.newaxis]
            
            # === the absorption by O3 === 
            i = self.n_O3*calc_of_effective(self.wl, self.xses_O3_293K)
            j = self.n_O3*calc_of_effective(self.wl_scat1, self.xses_O3_293K)
            k = self.n_O3*calc_of_effective(self.wl_scat2, self.xses_O3_293K)
            alpha_O3_s1 = i+j
            alpha_O3_s2 = i+k
            # alpha_s1 = alpha_mol_s1[np.newaxis, :] + alpha_aer_s1[np.newaxis, :] + alpha_SO2_s1 + alpha_H2S_s1 + alpha_O3_s1[np.newaxis, :]
            # alpha_s2 = alpha_mol_s2[np.newaxis, :] + alpha_aer_s2[np.newaxis, :] + alpha_SO2_s2 + alpha_H2S_s2 + alpha_O3_s2[np.newaxis, :]
            alpha_s1 = alpha_SO2_s1
            alpha_s2 = alpha_SO2_s2
           
            # いずれかの気体について吸収断面積の補間時にNaNが発生した波長列を破棄
            # idx_ok_s1 = ~np.isnan(alpha_s1).any(axis=0)
            # idx_ok_s2 = ~np.isnan(alpha_s2).any(axis=0)
            # self.wl = self.wl[idx_ok_s1&idx_ok_s2]
            # self.wl_scat1 = self.wl_scat1[idx_ok_s1&idx_ok_s2]
            # self.wl_scat2 = self.wl_scat2[idx_ok_s1&idx_ok_s2]
            # alpha_s1 = alpha_s1[:, idx_ok_s1&idx_ok_s2]
            # alpha_s2 = alpha_s2[:, idx_ok_s1&idx_ok_s2]
            # flg_s1_on = flg_s1_on[:, idx_ok_s1&idx_ok_s2]

            dr = int_R[1:] - int_R[:-1]
            alpha_s1 = (alpha_s1[1:, :] + alpha_s1[:-1, :]) * dr[:, np.newaxis] / 2
            alpha_s2 = (alpha_s2[1:, :] + alpha_s2[:-1, :]) * dr[:, np.newaxis] / 2
            # alpha_s1 = np.squeeze(alpha_s1[np.newaxis, :] * dr[:, np.newaxis])
            # alpha_s2 = np.squeeze(alpha_s2[np.newaxis, :] * dr[:, np.newaxis])

            # transmittance
            tau_s1 = np.exp(-np.cumsum(alpha_s1, axis=0))
            tau_s2 = np.exp(-np.cumsum(alpha_s2, axis=0))
            if self.way1: tau_s1 *= ratio_ASTK()
            if self.way2: tau_s2 *= ratio_ASTK()
            t1 = (
                self.E0
                *self.dR
                *self.A
                *self.eta
                *self.M
                *self.q
                /const.h
                *s_raman_N2
                *overlap(self.R[:, np.newaxis])
                /(self.R[:, np.newaxis]**2)
                *self.wl[np.newaxis, :]
                *1e-9
                /const.c
            )
            p_s1 = (
                tau_s1
                * t1
                * N(self.alt)
                * gases.at[self.scat1, "vol-ratio"]
                * (r_O2_raman if self.scat1 == "O2" else 1)
            )

            p_s2 = (
                tau_s2
                * t1
                * N(self.alt)
                * gases.at[self.scat2, "vol-ratio"]
                * (r_O2_raman if self.scat2 == "O2" else 1)
            )
            
            os.makedirs(os.getcwd()+self.out_dir, exist_ok=True)
            # p_s1.to_csv(os.getcwd()+self.out_dir+f"p_{self.scat1}-{"AS" if self.way1 else "ST"}.csv")
            # p_s2.to_csv(os.getcwd()+self.out_dir+f"p_{self.scat2}-{"AS" if self.way2 else "ST"}.csv")
            # meta = {
            #     "E0": [self.E0, float("nan")],  # 100 [mJ]
            #     "A": [self.A, float("nan")],
            #     "M": [self.M, float("nan")],  # 100 Hz / 1 hour
            #     "eta": [self.eta, float("nan")],
            #     "q": [self.q, float("nan")],
            #     "dR": [self.dR, float("nan")],
            #     "Bj": [self.Bj, float("nan")],
            #     "F": [self.F, float("nan")],
            #     "D": [self.D, float("nan")],
            #     "alt": [self.alt, float("nan")],
            #     "range_sim": self.range_sim,
            #     "range_dark_gas": self.range_dark_gas,
            #     "n_SO2": n_SO2,
            #     "n_H2S": n_H2S,
            #     "n_O3": self.n_O3,
            # }
            # ).to_csv(os.getcwd()+self.out_dir+f"meta.csv", index=False)
        else:
            p_s1 = pd.read_csv(
                f"{dir}/p_{self.scat1}-{"AS" if self.way1 else "ST"}.csv",
                header=0,
                index_col=0,
                dtype=float,
            )
            p_s2 = pd.read_csv(
                f"{dir}/p_{self.scat2}-{"AS" if self.way2 else "ST"}.csv",
                header=0,
                index_col=0,
                dtype=float,
            )
            meta = pd.read_csv(f"{dir}/meta.csv", header=0, index_col=None, dtype=float)
            self.E0 = meta["E0"][0]
            self.A = meta["A"][0]
            self.M = meta["M"][0]
            self.eta = meta["eta"][0]
            self.q = meta["q"][0]
            self.dR = meta["dR"][0]
            self.Bj = meta["Bj"][0]
            self.F = meta["F"][0]
            self.D = meta["D"][0]
            self.alt = meta["alt"][0]
            self.range_sim = meta["range_sim"]
            self.range_dark_gas = meta["range_dark_gas"]
            self.n_SO2 = meta["n_SO2"]
            self.n_H2S = meta["n_H2S"]
            self.n_O3  = meta["n_O3"]
            # self.alt = self.meta["alt"]
            if list(p_s1.index) == list(p_s2.index):
                self.wl = np.array(p_s1.columns.astype(float))
                self.R = np.array(p_s1.index.astype(float))
                self.aveR = (self.R[1:] + self.R[:-1]) / 2
            else:
                sys.exit("Error")
            p_s1 = p_s1.to_numpy()
            p_s2 = p_s2.to_numpy()
            self.wl_scat1 = wl_shift(self.wl, gases.at[self.scat1, "sft"], dir=self.way1)
            self.wl_scat2 = wl_shift(self.wl, gases.at[self.scat2, "sft"], dir=self.way2)
            i = calc_of_effective(self.wl_scat1, self.xses_SO2_358K)
            j = calc_of_effective(self.wl_scat1, self.xses_SO2_298K)
            xs_SO2_s1 = np.array([
                i if self.judge_dark(r) else j
                for r in self.R
            ])
            
            i = calc_of_effective(self.wl_scat2, self.xses_SO2_358K)
            j = calc_of_effective(self.wl_scat2, self.xses_SO2_298K)
            xs_SO2_s2 = np.array([
                i if self.judge_dark(r) else j
                for r in self.R
            ])
            flg_s1_on = xs_SO2_s1 > xs_SO2_s2

    #     if __MY_DEBUG_KEY__:

    #         fig1, ax1 = plt.subplots(1, 2, layout="tight")
    #         # ax.scatter(self.R_array, alpha_base[0, 1:])
    #         ax1[0].set_title(
    #             "distance@{:}m ".format(self.R_array[self.R_array.size - 2])
    #         )
    #         ax1[0].scatter(
    #             self.wl.index, T_S1[:, self.R_array.size - 2], label=f"{self.lb1} raman"
    #         )
    #         ax1[0].scatter(
    #             self.wl.index, T_S2[:, self.R_array.size - 2], label=f"{self.lb2} raman"
    #         )
    #         ax1[0].set_ylabel(r"laser wavelength [nm]")
    #         ax1[0].set_ylabel(r"$\tau$")
    #         # ax1[0].set_ylim(None, 1)
    #         ax1[0].set_yscale("log")
    #         ax1[0].legend()

    #         idx = serch_idx(316, self.wl.index)
    #         ax1[1].set_title("laser@{:}nm ".format(self.wl.index[idx]))
    #         ax1[1].plot(
    #             self.R_array, T_S1[idx, :], marker="o", label=f"{self.lb1} raman"
    #         )
    #         ax1[1].plot(
    #             self.R_array, T_S2[idx, :], marker="o", label=f"{self.lb2} raman"
    #         )
    #         ax1[1].set_ylabel(r"horizontal distance [m]")
    #         ax1[1].set_ylabel(r"$\tau$")
    #         # ax1[1].set_ylim(None, 1)
    #         ax1[1].set_yscale("log")
    #         ax1[1].legend()

    #         fig2, ax2 = plt.subplots(1, 2, layout="tight")
    #         # ax.scatter(self.R_array, alpha_base[0, 1:])
    #         ax2[0].set_title(
    #             "distance@{:}m ".format(self.R_array[self.R_array.size - 2])
    #         )
    #         ax2[0].scatter(
    #             self.wl.index,
    #             p_S1.iloc[:, self.R_array.size - 2],
    #             label=f"p_{self.lb1}",
    #         )
    #         ax2[0].scatter(
    #             self.wl.index,
    #             p_S2.iloc[:, self.R_array.size - 2],
    #             label=f"p_{self.lb2}",
    #         )
    #         ax2[0].set_ylabel(r"laser wavelength [nm]")
    #         ax2[0].set_ylabel(r"$p_{\text{phot}}$")
    #         ax2[0].set_yscale("log")
    #         ax2[0].legend()

    #         ax2[1].set_title("laser@{:}nm ".format(self.wl.index[idx]))
    #         ax2[1].plot(
    #             self.R_array, p_S1.iloc[idx, :], marker="o", label=f"p_{self.lb1}"
    #         )
    #         ax2[1].plot(
    #             self.R_array, p_S2.iloc[idx, :], marker="o", label=f"p_{self.lb2}"
    #         )
    #         ax2[1].set_ylabel(r"horizontal distance [m]")
    #         ax2[1].set_ylabel(r"$p_{\text{phot}}$")
    #         ax2[1].set_yscale("log")
    #         ax2[1].legend()

    #         plt.show(block=False)
    #         input()
    #         plt.close()

        return p_s1, p_s2, flg_s1_on

    def densities(self, d_xs, p_on, p_off):
        """
        _summary_

        Args:
            d_xs_SO2 (_dict[int, pd.Dataframe]_): _description_
            p_on (_pd.Dataframe_): _description_
            p_off (_pd.Dataframe_): _description_
            update (_bool_): _description_. Defaults to True.

        Returns:
            pd.Dataframe:
            n1, the density without corr
        """
        t1 = 1 / d_xs / self.dR /2
        t2 = np.log((p_off[1:, :] * p_on[:-1, :]) / (p_on[1:, :] * p_off[:-1, :]))

        return t1 * t2

    def stdev_densitis(
        self,
        d_xs,
        p_on,
        p_off,
    ):
        """Statistical error solve"""
        t1 = 1 / d_xs / self.dR
        s_on = (p_on + self.Bj) * self.F + self.D
        s_off = (p_off + self.Bj) * self.F + self.D

        t2_on = s_on / np.power(p_on, 2)
        t2_off = s_off / np.power(p_off, 2)

        t2 = t2_on[1:, :] + t2_on[:-1, :] + t2_off[1:, :] + t2_off[:-1, :]
        t2 = np.power(t2, 1 / 2)

        return t1 * t2

    def print_param(self):
        print("=============================================================")
        print("altitude [m]: {}".format(self.alt))
        print(
            "dark gas [m]: {} - {}".format(
                self.range_dark_gas[0], self.range_dark_gas[1]
            )
        )
        print(
            "R [m]       : {:<5.0f}, {:<5.0f}, ..., {:<5.0f} / {:>3}".format(
                self.R_array[0],
                self.R_array[1],
                self.R_array[-1],
                len(self.R_array),
            )
        )

        print(
            "\n{:<14}: {:<8.3f}, {:<8.3f}, ..., {:<8.3f} [nm]~ {:>3}".format(
                "wl",
                self.wl.index[0],
                self.wl.index[1],
                self.wl.index[-1],
                len(self.wl.index),
            )
        )

        print(
            "{:<14}: {:<8.3f}, {:<8.3f}, ..., {:<8.3f} [nm]~ {:>3}".format(
                "O2 ANTI_STOKES" if self.way1 else "O2 STOKES",
                self.wl[self.lb1].to_numpy()[0],
                self.wl[self.lb1].to_numpy()[1],
                self.wl[self.lb1].to_numpy()[-1],
                len(self.wl.index),
            )
        )
        print(
            "{:<14}: {:<8.3f}, {:<8.3f}, ..., {:<8.3f} [nm]~ {:>3}".format(
                "N2 ANTI_STOKES" if self.way2 else "N2 STOKES",
                self.wl[self.lb2].to_numpy()[0],
                self.wl[self.lb2].to_numpy()[1],
                self.wl[self.lb2].to_numpy()[-1],
                len(self.wl.index),
            )
        )
        print("element  : {}".format(len(self.wl.index) * len(self.R_array)))

        print("\npulse enegey : {} [mJ]".format(self.E0 * 1000))
        print("diameter     : {} [m]".format(self.A))
        print("cum. shot    : {:.2g}".format(self.M))
        print("eff. detec.  : {}".format(self.eta))
        print("eff. optic.  : {}".format(self.q))
        print("dR           : {} [m]".format(self.dR))
        print("nois bg      : {}".format(self.Bj))
        print("nois detec.  : {}".format(self.F))
        print("dark count   : {}".format(self.D))

        print()
        print("density O3  : {:.3g} [ppm]".format(self.DENSITY_O3 * 1e6))
        print(
            "density SO2 : (thin) {:.3g} / (thick) {:.3g} [ppm]".format(
                self.DENSITY_SO2[0] * 1e6, self.DENSITY_SO2[1] * 1e6
            )
        )
        print(
            "density H2S : (thin) {:.3g} / (thick) {:.3g} [ppm]".format(
                self.DENSITY_H2S[0] * 1e6, self.DENSITY_H2S[1] * 1e6
            )
        )
        print()
        print("estimated density O3  : {:.3g} [ppm]".format(self.n_O3_est * 1e6))
        print("estimated density H2S : {:.3g} [ppm]".format(self.n_H2S_est * 1e6))
        print("=============================================================")

    def main(self, new=False):
        """受信光子数計算部"""
        ft_aer_true = 1
        ft_aer_est = 1

        n_SO2_est = N(self.alt)*30*1e-6
        n_H2S_est = N(self.alt)*45*1e-6
        n_O3_est = N(self.alt)*0.005*1e-6

        p_s1, p_s2, flg_s1_on = self.powers(ft_aer_true, new=new)
        p_on, p_off = onoff_swapper(flg_s1_on, p_s1, p_s2)

        # self.p_on = pd.DataFrame(data=p_on,   index=self.R, columns=self.wl)
        # self.p_off = pd.DataFrame(data=p_off, index=self.R, columns=self.wl)

        """ 波長依存性補正項計算部"""
        # d_xs_SO2
        i = calc_of_effective(self.wl_scat1, self.xses_SO2_358K)
        j = calc_of_effective(self.wl_scat1, self.xses_SO2_298K)
        xs_SO2_s1 = np.array([
            i if self.judge_dark(r) else j
            for r in self.aveR
        ])
        
        i = calc_of_effective(self.wl_scat2, self.xses_SO2_358K)
        j = calc_of_effective(self.wl_scat2, self.xses_SO2_298K)
        xs_SO2_s2 = np.array([
            i if self.judge_dark(r) else j
            for r in self.aveR
        ])
        flg_s1_on = xs_SO2_s1 > xs_SO2_s2
        xs_SO2_on, xs_SO2_off = onoff_swapper(flg_s1_on, xs_SO2_s1, xs_SO2_s2)
        d_xs = xs_SO2_on-xs_SO2_off

        # d_alpha_mol
        alpha_mol_s1 = calc_of_effective(self.wl_scat1, alphas_mol, args=(self.alt))
        alpha_mol_s2 = calc_of_effective(self.wl_scat2, alphas_mol, args=(self.alt))
        alpha_mol_s1, _ = np.meshgrid(alpha_mol_s1, self.aveR)
        alpha_mol_s2, _ = np.meshgrid(alpha_mol_s2, self.aveR)
        alpha_mol_on, alpha_mol_off = onoff_swapper(flg_s1_on, alpha_mol_s1, alpha_mol_s2)
        self.d_alpha_mol = alpha_mol_on - alpha_mol_off

        # d_alpha_aer
        alpha_aer_s1 = calc_of_effective(self.wl_scat1, alphas_aer, args=(ft_aer_est,self.alt))
        alpha_aer_s2 = calc_of_effective(self.wl_scat2, alphas_aer, args=(ft_aer_est,self.alt))
        alpha_aer_s1, _ = np.meshgrid(alpha_aer_s1, self.aveR)
        alpha_aer_s2, _ = np.meshgrid(alpha_aer_s2, self.aveR)
        alpha_aer_on, alpha_aer_off = onoff_swapper(flg_s1_on, alpha_aer_s1, alpha_aer_s2)
        self.d_alpha_aer = alpha_aer_on - alpha_aer_off

        # d_alpha_H2S
        i = calc_of_effective(self.wl_scat1, self.xses_H2S_423K)
        j = calc_of_effective(self.wl_scat1, self.xses_H2S_294K)
        xs_H2S_s1 = np.array([
            i if self.judge_dark(r) else j
            for r in self.aveR
        ])
        i = calc_of_effective(self.wl_scat2, self.xses_H2S_423K)
        j = calc_of_effective(self.wl_scat2, self.xses_H2S_294K)
        xs_H2S_s2 = np.array([
            i if self.judge_dark(r) else j
            for r in self.aveR
        ])
        xs_H2S_on, xs_H2S_off = onoff_swapper(flg_s1_on, xs_H2S_s1, xs_H2S_s2)
        self.d_alpha_H2S = n_H2S_est*(xs_H2S_on-xs_H2S_off)

        i = calc_of_effective(self.wl_scat2, self.xses_H2S_423K)
        j = calc_of_effective(self.wl_scat2, self.xses_H2S_294K)
        xs_O3_s1, _ = np.meshgrid(calc_of_effective(self.wl_scat1, self.xses_O3_293K), self.aveR)
        xs_O3_s2, _ = np.meshgrid(calc_of_effective(self.wl_scat2, self.xses_O3_293K), self.aveR)
        xs_O3_on, xs_O3_off = onoff_swapper(flg_s1_on, xs_O3_s1, xs_O3_s2)
        self.d_alpha_O3 = n_O3_est*(xs_O3_on-xs_O3_off)
        
        """ 気体濃度分布計算部 """
        self.n1 = pd.DataFrame(
            data=self.densities(d_xs, p_on, p_off),
            index=self.aveR,
            columns=self.wl,
        )
        self.d_alpha_all = (
            self.d_alpha_mol + self.d_alpha_aer + self.d_alpha_H2S + self.d_alpha_O3
        )

        self.n3 = self.n1 - (self.d_alpha_all / d_xs)

        self.stdev_n = pd.DataFrame(
            data=self.stdev_densitis(d_xs, p_on, p_off),
            index=self.aveR,
            columns=self.wl,
        )
        # self.n_rand = pd.DataFrame(
        #     data=np.array([np.random.normal(0, i) for i in self.stdev_n.to_numpy()]).reshape(self.stdev_n.shape),
        #     index=self.wl,
        #     columns=self,aveR
        # )
        self.n_true = self.dist_SO2(self.R)
        self.n_true = (self.n_true[1:]+self.n_true[:-1])/2
        self.n_true = self.n_true[:, np.newaxis]

        self.error1 = 100 * (self.n1 - self.n_true) / self.n_true
        self.error3 = 100 * (self.n3 - self.n_true) / self.n_true
        self.stdev_error = 100 * self.stdev_n / self.n_true
        # self.total_error = 100 * (self.n3 + self.n_rand - self.n_true) / self.n_true

        # elapsed_time = perf_counter() - self.start
        # end = strftime("%H:%M:%S", gmtime(elapsed_time))


if __name__ == "__main__":
    print("test simulation ...\n")
    sim = rDIAL_sim_package("O2", False, "N2", False)
    sim.entry_param(dR=50)
    # p = sim.powers(1, new=True)
    sim.main(new=True)
    fig, axes = plt.subplots(2, 2)
    
    for ax in axes.ravel():
        ax.grid(which="major", c="darkgrey", ls="-")
        ax.grid(which="minor", c="lightgrey", ls="--")
        ax.set_axisbelow(True)
        ax.set_yscale("log")

    for ax in axes[0, :]: ax.set_ylabel("SO2 error(other interference)[%]")
    for ax in axes[1, :]: ax.set_ylabel("statistical error[%]")
    for ax in axes[:, 0]: ax.set_xlabel("laser wavelength [nm]")
    for ax in axes[:, 1]: ax.set_xlabel("horizontal distance [m]")
    idx = sim.error1.iloc[-1, :].idxmin()
    # sim.n1 /= N(sim.alt)
    axes[0, 0].scatter(sim.wl,   sim.error1.iloc[-5, :].abs())
    axes[0, 1].scatter(sim.aveR, sim.dist_SO2(sim.aveR))
    axes[0, 1].scatter(sim.aveR, sim.n1.loc[:, sim.error1.iloc[-1, :].idxmin()].abs())
    axes[1, 0].scatter(sim.wl,   sim.stdev_error.iloc[-1, :].abs())
    axes[1, 1].scatter(sim.aveR, sim.stdev_error.loc[:, sim.stdev_error.iloc[-1, :].idxmin()].abs())
    plt.show(block=False)
    input()    
