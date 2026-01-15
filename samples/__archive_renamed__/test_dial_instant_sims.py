# coding: utf-8
import sys
import numpy as np
from dataclasses import replace
from numpy.lib.stride_tricks import sliding_window_view as np_SWV
from matplotlib import pyplot as plt
from pathlib import Path

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from gas_simulation import utils
from gas_simulation.consts import main_gases_props
from gas_simulation.atom import betas_N2, betas_O2

from gas_simulation.model import Gas, InstantEnvironment
from gas_simulation.lidar_model.utils import Coord
from gas_simulation.lidar_model import lidar
from gas_simulation.lidar_model import dial
from gas_simulation import result_viewer as viewer

xs_SO2 = utils.load_cross_section(
    "SO2_VandaeleHermansFally(2009)_358K_227.275-416.658nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)
xs_H2S = utils.load_cross_section(
    "H2S_Grosch(2015)_423.2K_198-370nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)
xs_O3 = utils.load_cross_section(
    "O3_Bogumil(2003)_293K_230-1070nm.xlsx",
    interp_kwargs={"bounds_error": False, "fill_value": np.nan},
    effective=True,
)

t_sec = 60 * 15  # [sec]
lc = lidar.LidarCalc(M=100 * t_sec)
dc = dial.DialCalc()

lidar_coord = Coord(
    distance=np.arange(5, 1000, 5),
    theta_deg=0.0,
    x0=0,
    z0=1000,
)

env = InstantEnvironment(
    gas={
        # 煙源直下で30ppmになるよう調整したQ
        # "SO2": Gas(Q=80e5, offset=0, cross_section=xs_SO2),
        # "H2S": Gas(Q=40e5, offset=0, cross_section=xs_H2S),
        # 今回の煙源位置、風プロファイル設定で30ppm程度になるよう調整したQ
        "SO2": Gas(Q=30, offset=0.07, cross_section=xs_SO2),
        "H2S": Gas(Q=15, offset=0.035, cross_section=xs_H2S),
        "O3": Gas(Q=0, offset=0.005, cross_section=xs_O3),
    },
    time=t_sec,
    trig=lambda x: ((x >= 400) & (x <= 600)),
)
env.show_gases(lidar_coord)

wl_laser = np.arange(240, 370, 0.02)
wl_s1 = utils.wl_shift(wl_laser, main_gases_props.at["N2", "sft"], True)
wl_s2 = utils.wl_shift(wl_laser, main_gases_props.at["O2", "sft"], True)

beta_N2 = betas_N2(lidar_coord.z) * 0.1
beta_O2 = betas_O2(lidar_coord.z) * 0.1

tau_laser = env.transmittance(lidar_coord, wl_laser)
tau_s1 = tau_laser * env.transmittance(lidar_coord, wl_s1)
tau_s2 = tau_laser * env.transmittance(lidar_coord, wl_s2)

# === power calculation ===
p_s1 = lc.power(
    dist=lidar_coord.distance[:, np.newaxis],
    wl=wl_laser[np.newaxis, :],
    beta_tau=beta_N2[:, np.newaxis] * tau_s1,
)
p_s2 = lc.power(
    dist=lidar_coord.distance[:, np.newaxis],
    wl=wl_laser[np.newaxis, :],
    beta_tau=beta_O2[:, np.newaxis] * tau_s2,
)

obj_s1 = dial.RamanShiftObject(
    wl_s1, env.gas_inventory["SO2"].cross_section(wl_s1), p_s1
)
obj_s2 = dial.RamanShiftObject(
    wl_s2, env.gas_inventory["SO2"].cross_section(wl_s2), p_s2
)
dial_input = dial.DialInput(
    env=env,
    obj_s1=obj_s1,
    obj_s2=obj_s2,
    lidar_coord=lidar_coord,
    sumN=1,
)
results, cf_input, debug = dc.estimate(dial_input)
cf = dial.calc_correction_factor(cf_input)
debug.tau_on = np.where(debug.mask[np.newaxis, :], tau_s1, tau_s2)
debug.tau_off = np.where(debug.mask[np.newaxis, :], tau_s2, tau_s1)

idx_trgt = np.nanargmin(results.stat_err[-1, :])
# idx_trgt = np.searchsorted(wl_laser, 320)

# === test plot ===
fig1, axes1 = viewer.lidar_equation_result_viewer(
    lidar_coord=lidar_coord,
    wl=wl_laser,
    debug=debug,
    dist_idx=-1,
    wl_idx=idx_trgt,
)
plt.show(block=False)

fig2, axes2 = viewer.dial_equation_result_viewer(
    env=env, wl=wl_laser, results=results, cf = cf, dist_idx=-1, wl_idx=idx_trgt
)
plt.show(block=False)

# 信号処理変更
dial_input = replace(dial_input, sumN=5)
results, cf_input, debug = dc.estimate(dial_input)
cf = dial.calc_correction_factor(cf_input)
print(results.n_true["SO2"].shape)
print(results.coord.z.shape)
idx_trgt = np.nanargmin(100 * results.stat_err[-1, :] / results.n_true["SO2"][-1])
print(wl_laser[idx_trgt])
fig3, axes3 = viewer.dial_equation_result_viewer(
    env=env, wl=wl_laser, results=results, cf = cf, dist_idx=-1, wl_idx=idx_trgt
)

# res = dc.calc(
#     p_on_R1=p_on_R1,
#     p_on_R2=p_on_R2,
#     p_off_R1=p_off_R1,
#     p_off_R2=p_off_R2,
#     dR=dial_dR[:, np.newaxis],
#     d_xs=d_xs_SO2[np.newaxis, :],
# )
# stat_err = dc.stat_error(
#     p_on_R1=p_on_R1,
#     p_on_R2=p_on_R2,
#     p_off_R1=p_off_R1,
#     p_off_R2=p_off_R2,
#     dR=dial_dR[:, np.newaxis],
#     d_xs=d_xs_SO2[np.newaxis, :],
# )
# dial_correction_factor = dial.calc_dial_correction_factor(
#     env=env,
#     alt=dial_coord.z[:, np.newaxis],
#     wl_on=wl_on[np.newaxis, :],
#     wl_off=wl_off[np.newaxis, :],
#     d_xs=d_xs_SO2[np.newaxis, :],
# )

# fig3, axes3 = viewer.dial_equation_result_viewer(
#     env=env,
#     coord=dial_coord,
#     n_true_dist=n_true["SO2"],
#     res1_dist=res[:, idx_320nm],
#     res2_dist=(res - dial_correction_factor)[:, idx_320nm],
#     stat_err_dist=stat_err[:, idx_320nm],
#     wl=wl_laser,
#     n_true_wl=n_true["SO2"][-1],
#     res1_wl=res[-1, :],
#     res2_wl=(res - dial_correction_factor)[-1, :],
#     stat_err_wl=stat_err[-1, :],
# )

plt.show(block=False)
input("PRESS ANY KEY...")
