# coding: utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gas_simulation.utils import gaus
from gas_simulation.utils import effective
from gas_simulation.utils import load_cross_section

x = np.linspace(-2, 2, 1000)
y = gaus(x, mean=0, FWHM=1, normalize="peak")
y = y/y.sum()

t = np.linspace(-100, 100, 1000)[:, np.newaxis, np.newaxis]
freq1 = 1/40
freq2 = 4

f = lambda x: np.sin(2*np.pi*freq1*x) + np.sin(2*np.pi*freq2*x)/10
I = f(t)
I_eff = effective(t, f)

fig, axes = plt.subplots(1,2)
for ax in axes:
  ax.grid(which="minor", ls="--", c="lightgrey")
  ax.grid(which="major", ls="-", c="darkgrey")
axes[0].plot(x, y)
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[1].plot(t.ravel(), I.ravel(), label="inst")
axes[1].plot(t.ravel(), I_eff.ravel(), label="eff")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Intensity")
axes[1].legend()
plt.show(block=False)

wl = np.linspace(230, 370, 5000)
xs_SO2_1 = load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan}, 
)

xs_SO2_2 = load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx", 
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True
)
fig2, ax2 = plt.subplots(1,1)
ax2.grid(which="minor", ls="--", c="lightgrey")
ax2.grid(which="major", ls="-", c="darkgrey")
ax2.plot(wl, xs_SO2_1(wl), label="inst")
ax2.plot(wl, xs_SO2_2(wl), label="eff")
ax2.set_xlabel("wavelength [nm]")
ax2.set_ylabel("cross section")
ax2.legend()
plt.show(block=False)

input("ENTER ANY KEY......")
