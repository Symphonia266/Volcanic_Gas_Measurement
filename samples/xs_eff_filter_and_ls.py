# ================================
# Standard library imports
# ================================
import sys
from pathlib import Path
from dataclasses import dataclass

# ================================
# Third-party imports
# ================================
import numpy as np
import pandas as pd

from matplotlib import cm, ticker
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter

from numpy.lib.stride_tricks import sliding_window_view as np_SWV
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ================================
# Project-specific imports
# ================================
import config as cfg
from gas_simulation import utils
from gas_simulation.consts import main_gases_props
from gas_simulation.atom import betas_N2, betas_O2
from gas_simulation.diffusion_model import pasquill_stable_classfication as PSC

from gas_simulation.lidar_model.utils import Coord
from gas_simulation.lidar_model import lidar
from gas_simulation.lidar_model import dial
from gas_simulation import result_viewer as viewer

# ================================
# Matplotlib global style
# ================================
plt.style.use(cfg.MPLSTYLE_PATH)
cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# Cross sections
# ================================
xses_SO2 = utils.load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=False,
)
xses_H2S = utils.load_cross_section(
    "H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=False,
)
xses_O3 = utils.load_cross_section(
    "O3_Bogumil(2003)_293K_230-1070nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=False,
)

xses_SO2_eff = utils.load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)

if __name__ == "__main__":

    # レーザスペクトル設定
    # a = np.linspace(240, 370)
    # b = {
    #     "N2_as" : utils.wl_shift(a, main_gases_props.at["N2", "sft"], True),
    #     "O2_as" : utils.wl_shift(a, main_gases_props.at["O2", "sft"], True),
    #     "ls" : a, 
    #     "O2_st" : utils.wl_shift(a, main_gases_props.at["O2", "sft"], False),
    #     "N2_st" : utils.wl_shift(a, main_gases_props.at["N2", "sft"], False)
    # } 
    # diff_1 = b["O2_as"] - b["N2_as"]
    # diff_2 = b["ls"] - b["O2_as"]
    # diff_3 = b["O2_st"] - b["ls"]
    # diff_4 = b["N2_st"] - b["O2_st"]
    # plt.figure()
    # plt.plot(a, diff_1, label="diff_1")
    # plt.plot(a, diff_2, label="diff_2")
    # plt.plot(a, diff_3, label="diff_3")
    # plt.plot(a, diff_4, label="diff_4")
    # plt.legend()
    # plt.show(block=False)
    # input()

    wl_spec = np.linspace(-2, 2, 500)
    spec = lambda x, wl_cnt : utils.gaus(x, mean=wl_cnt, FWHM=1, normalize="peak")
  
    wl_ls = wl_spec + 334.6
    wls = {
        "N2_as" : utils.wl_shift(wl_ls, main_gases_props.at["N2", "sft"], True),
        "O2_as" : utils.wl_shift(wl_ls, main_gases_props.at["O2", "sft"], True),
        "ls"    : wl_ls, 
        "O2_st" : utils.wl_shift(wl_ls, main_gases_props.at["O2", "sft"], False),
        "N2_st" : utils.wl_shift(wl_ls, main_gases_props.at["N2", "sft"], False)
    } 
    wl = np.concatenate(list(wls.values()))
    
    cnt_as = wls["O2_as"].mean()
    cnt_st = wls["O2_st"].mean()
    wdt_as = 13
    wdt_st = 15
    flt = lambda wl, cnt, width: utils.gaus(wl, mean=cnt, FWHM=width, normalize="peak")    
    flt_as = lambda wl: flt(wl, cnt_as, wdt_as)*flt(wl, cnt_as, wdt_as)
    flt_st = lambda wl: flt(wl, cnt_st, wdt_st)*flt(wl, cnt_st, wdt_st)

    Amp = np.concatenate([
        0.1 * spec(wl_spec, 0),
        0.1 * spec(wl_spec, 0),
        spec(wl_spec, 0),
        spec(wl_spec, 0),
        spec(wl_spec, 0),
    ])
    Amp_flt_st = Amp*0.9*0.9*flt_st(wl)
    Amp_flt_as = Amp*0.9*0.9*flt_as(wl)

    I = np.tile(spec(wl_spec, 0), 5)
    I_as = I*flt_as(wl)
    I_as /= I_as.sum()
    I_st = I*flt_st(wl)
    I_st /= I_st.sum()

    xs_SO2_eff_chON  = (I_as*xses_SO2(wl)).sum()
    xs_SO2_eff_chOFF = (I_st*xses_SO2(wl)).sum()
  
    xs_SO2_on = xses_SO2_eff(wls["O2_as"].mean())
    xs_SO2_off = xses_SO2_eff(wls["O2_st"].mean())

    print(xs_SO2_eff_chON)
    print(xs_SO2_on)
    print(xs_SO2_eff_chOFF)
    print(xs_SO2_off)

    # xs_H2S = (A_flt_as*xses_H2S(wl)).sum()
    # xs_O3 = (A_flt_as*xses_O3(wl)).sum()

    fig, ax = plt.subplots()
    axins = ax.inset_axes([0.2, 0.2, 0.4, 0.6])
    # ax.plot(wl, A, c="black", label="")
    ax.plot(wl, Amp_flt_as, ls="-", label="on channel")
    ax.plot(wl, Amp_flt_st, ls="--", label="off channel")
    axins.plot(wl, Amp_flt_as, ls="-", label="on channel")
    ax.indicate_inset_zoom(axins)
    
    x = np.linspace(240, 370, 1000)
    fig2, ax2 = plt.subplots()
    ax2.plot(x, 0.9*0.9*flt_as(x), c="black", ls="-", label="a-Stokes side")
    ax2.plot(x, 0.9*0.9*flt_st(x), c="black", ls="--", label="Stokes side")

    y = (
        spec(x, wls["N2_as"].mean())+
        spec(x, wls["O2_as"].mean())+
        spec(x, wls["ls"].mean())+
        spec(x, wls["O2_st"].mean())+
        spec(x, wls["N2_st"].mean())
    )
    fig3, ax3 = plt.subplots()
    ax3.plot(x, xses_SO2(x),     ls="-" , c="black", label="raw")
    ax3.plot(x, xses_SO2_eff(x), ls="--", c="grey" , label="eff")
    # ax3.plot(x, flt_as(x)*xses_SO2(x), ls="--", c="blue" , label="on ch.")
    # ax3.plot(x, flt_st(x)*xses_SO2(x), ls="--", c="orange" , label="off ch.")
    # ax3.plot(x, y, label="spec")
    # ax3.plot(x, y*flt_as(x), label="on ch.")
    # ax3.plot(x, y*flt_st(x), label="off ch.")
    
    def draw_fwhm_arrow(ax, center, fwhm, y, *, label=None):
        x1 = center - fwhm / 2
        x2 = center + fwhm / 2

        # 縦線（半値位置）
        ax.vlines([x1, x2], 0, y, colors="red", linestyles=":")

        # 両端矢印
        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="<->",
                color="red",
            ),
        )
        ax.annotate(
            f"{fwhm:.1f} nm",
            xy=(center, y),
            xytext=(0, -5),          # 上方向に 5 pt
            textcoords="offset points",
            ha="center",
            va="top",
            c="red",
        )

        # ラベル
        if label is not None:
            ax.text(
                center, y,
                label,
                ha="center",
                va="bottom",
            )
    ax2.axhline(0.9*0.9, 0, 1, ls="-", c="red")
    ax2.text(0.01, 0.9*0.9, f"{100*0.9*0.9} %", c="red", va="bottom", ha="left",transform=ax2.transAxes)
    draw_fwhm_arrow(ax2, cnt_st, wdt_st/np.sqrt(2), 0.9*0.9*0.5)
    draw_fwhm_arrow(ax2, cnt_as, wdt_as/np.sqrt(2), 0.9*0.9*0.5)
    # ax.hlines(0.9*0.9*0.5, cnt_as-wdt_as/2, cnt_as+wdt_as/2, colors="gray")
    # ax.hlines(0.9*0.9*0.5, cnt_as-wdt_as/2, cnt_as+wdt_as/2, colors="gray")

    # 凡例
    ax.legend(    
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        loc="upper left",
    )
    ax2.legend(    
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        loc="upper left",
    )
    ax3.legend(    
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        loc="upper left",
    )

    # 軸設定
    ax.set(
        xlabel="Wavelength [nm]",
        ylabel="Amplitude",
        xlim=(wl.min(), wl.max()),
        ylim=(0, 1),
    )
    axins.set(
        xlim=(wl.min(), 320),
        ylim=(0, 0.09),
    )
    axins.set_title("on channel (zoom)", fontsize=9)
    # ax.yaxis.set_major_locator(LogLocator(base=10))
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    ax2.set(
        xlabel="Wavelength [nm]",
        ylabel="Filter efficiency",
        xlim=(wl.min(), wl.max()),
        ylim=(0, 1)
    )
    ax3.set(
        xlabel="Wavelength [nm]",
        ylabel=r"SO2 absorp cross section[m$^2$]",
        xlim=(wl.min(), wl.max()),
        ylim=(0, None)
    )

    # fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    # fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    # fig3.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.savefig(
        fname=str(cfg.OUT_DIR / f"sim_05_Amplitude.{cfg.EXT}"), 
        format=cfg.EXT
    )
    fig2.savefig(
        fname=str(cfg.OUT_DIR / f"sim_05_Filter_Spec.{cfg.EXT}"), 
        format=cfg.EXT
    )
    fig3.savefig(
        fname=str(cfg.OUT_DIR / f"sim_05_SO2_Spec.{cfg.EXT}"), 
        format=cfg.EXT
    )
    plt.show(block=False)
    input()

    print("< filtered (for stokes) power >")
    print("N2-st  : {:.1f} [%]".format(0.9**2*flt_st(wls["N2_st"].mean()) * 100))
    print("O2-st  : {:.1f} [%]".format(0.9**2*flt_st(wls["O2_st"].mean()) * 100))
    print("base   : {:.1f} [%]".format(0.9**2*flt_st(wls["ls"].mean()) * 100))
    print("O2-ast : {:.1f} [%]".format(0.9**2*flt_st(wls["O2_as"].mean()) * 100))
    print("N2-ast : {:.1f} [%]".format(0.9**2*flt_st(wls["N2_as"].mean()) * 100))

    print("< filtered (for a-stokes) power >")
    print("N2-st  : {:.1f} [%]".format(0.9**2*flt_as(wls["N2_st"].mean()) * 100))
    print("O2-st  : {:.1f} [%]".format(0.9**2*flt_as(wls["O2_st"].mean()) * 100))
    print("base   : {:.1f} [%]".format(0.9**2*flt_as(wls["ls"].mean()) * 100))
    print("O2-ast : {:.1f} [%]".format(0.9**2*flt_as(wls["O2_as"].mean()) * 100))
    print("N2-ast : {:.1f} [%]".format(0.9**2*flt_as(wls["N2_as"].mean()) * 100))
    
    print(f"on  : {xs_SO2_on:.3g} m2")
    print(f"on  : {xs_SO2_eff_chON:.3g} m2")
    print(f"off : {xs_SO2_off:.3g} m2")
    print(f"off : {xs_SO2_eff_chOFF:.3g} m2")
