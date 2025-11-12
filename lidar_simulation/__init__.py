import re
import os
import json
from matplotlib import pyplot as plt
from cycler import cycler

from func.io_utils import to_rc_dict

plt.rcParams["axes.prop_cycle"] = cycler(
    "color",
    ["#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF", "#F6AA00", "#FFF100", "#000000"],
)
with open("rcParams.json") as f:
    plt.rcParams.update(to_rc_dict(json.load(f)))

package_path = "".join(re.split(r"(\\)", __file__)[:-1])
cmap = [
    "#FF4B00",  # R255 G75 B0 鮮やかな黄みの赤
    "#005AFF",  # R0 G90 B255	鮮やかな青
    "#03AF7A",  # R3 G175 B122	重厚な青みの緑
    "#4DC4FF",  # R77 G196 B255	シアン色
    "#F6AA00",  # R246 G170 B0	鮮やかな黄みの橙
    "#FFF100",  # R255 G241 B0	鮮やかな黄
    "#000000",  # R0 G0 B0	漆黒の黒色
]
# def generate_cmap(colors, cmap_name="custom_cmap"):
#     values = range(len(colors))
#     vmax = np.ceil(np.max(values))
#     color_list = []
#     for vi, ci in zip(values, colors):
#         color_list.append((vi / vmax, ci))

#     return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)


# cmthermal = generate_cmap(
#     ["#1c3f75", "#068fb9", "#f1e235", "#d64e8b", "#730e22"], "cmthermal"
# )

# 他モジュールから import 可能にする
__all__ = ["package_path"]
__MY_DEBUG_KEY__ = False
