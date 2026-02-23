import numpy as np
import pandas as pd
from . import utils
from .consts import main_gases_props

def wls_setup(wl_ls):
  wl = {
      "laser": wl_ls,
      "N2_st": utils.wl_shift(wl_ls, main_gases_props.at["N2", "sft"], False),
      "O2_st": utils.wl_shift(wl_ls, main_gases_props.at["O2", "sft"], False),
      "N2_as": utils.wl_shift(wl_ls, main_gases_props.at["N2", "sft"], True),
      "O2_as": utils.wl_shift(wl_ls, main_gases_props.at["O2", "sft"], True),
  }
  return wl

def xses_setup(eff=False):
  xs_SO2 = utils.load_cross_section(
      "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
      interp_kwargs={"bounds_error": False, "fill_value": np.nan},
      effective=eff,
  )
  xs_H2S = utils.load_cross_section(
      "H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
      interp_kwargs={"bounds_error": False, "fill_value": np.nan},
      effective=eff,
  )
  xs_O3 = utils.load_cross_section(
      "O3_Bogumil(2003)_293K_230-1070nm.xlsx",
      interp_kwargs={"bounds_error": False, "fill_value": np.nan},
      effective=eff,
  )
  return xs_SO2, xs_H2S, xs_O3
