import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cycler import cycler

def to_rc_dict(dict):
    return {f"{k1}.{k2}": v for k1, d in dict.items() for k2, v in d.items()}

plt.rcParams["axes.prop_cycle"] = cycler(
    "color",
    ["#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF", "#F6AA00", "#FFF100", "#000000"],
)
with open("rcParams.json") as f:
    plt.rcParams.update(to_rc_dict(json.load(f)))

# generating JIS color pallet
cmap = [
    "#FF4B00",  # R255 G75 B0 鮮やかな黄みの赤
    "#005AFF",  # R0 G90 B255	鮮やかな青
    "#03AF7A",  # R3 G175 B122	重厚な青みの緑
    "#4DC4FF",  # R77 G196 B255	シアン色
    "#F6AA00",  # R246 G170 B0	鮮やかな黄みの橙
    "#FFF100",  # R255 G241 B0	鮮やかな黄
    "#000000",  # R0 G0 B0	漆黒の黒色
]
"""
universal color 7 map
---
::

      ---   R   G   B
  [0] ---  255  75   0   鮮やかな黄みの赤   #FF4B00  
  [1] ---    0  90 255   鮮やかな青        #005AFF 
  [2] ---    3 175 122   重厚な青みの緑     #03AF7A 
  [3] ---   77 196 255   シアン色          #4DC4FF 
  [4] ---  246 170   0   鮮やかな黄みの橙   #F6AA00 
  [5] ---  255 241   0   鮮やかな黄        #FFF100 
  [6] ---    0   0   0   漆黒の黒色        #000000 
"""

xs_SO2_358K_raw = pd.read_excel(
    f"SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
    index_col=0,
)
xs_O3_293K_raw = pd.read_excel(
    f"O3_Bogumil(2003)_293K_230-1070nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
    index_col=0,
)
xs_H2S_423K_raw = pd.read_excel(
    f"H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
    # sheet_name=None,
    header=4,
    names=["WL[nm]", "XS[m2/mol]"],
    index_col=0,
)
a=10*1e-6
b=1.5*1e-6
c=1*1e-6

fig, ax = plt.subplots(1, 1, layout="tight")
# ax.grid()
ax.plot(xs_SO2_358K_raw.index, a*xs_SO2_358K_raw, c=cmap[0], label=f"SO2={a*1e6:.2g}[ppm]")
ax.plot(xs_H2S_423K_raw.index, b*xs_H2S_423K_raw, c=cmap[1], label=f"H2S={b*1e6:.2g}[ppm]")
ax.plot(xs_O3_293K_raw.index, c*xs_O3_293K_raw, c=cmap[2], label=f"O3={c*1e6:.2g}[ppm]")
ax.set_xlabel("wavelength [nm]")
ax.set_ylabel(r"$\alpha$ [m$^{-1}$]")
ax.set_xlim(198, 350)
ax.legend()
plt.show(block=False)
input()
