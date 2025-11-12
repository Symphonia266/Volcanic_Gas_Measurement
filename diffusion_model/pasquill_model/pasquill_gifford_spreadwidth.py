import os
import numpy as np
import pandas as pd
# from scipy.interpolate import interp1d
from functools import partial
from scipy.interpolate import interp1d 

from . import package_path
from ..pasquill_stable_classfication import (
    classification_table,
    classification_label,
    classify_atomosphere_stability,
)
# from ..func import correct_time
from ..func import connect_smoothly_multi

def func(x, a, b):
    return (10**b)*(x**a)

def loglog_interp(x, y, **kwargs):
    logx = np.log10(x)
    logy = np.log10(y)
    f_log = interp1d(logx, logy, **kwargs)

    def f(x_new):
        return 10 ** f_log(np.log10(x_new))

    return f

def loglog_power_fit(x, y, mask):
    """In the mask's true interval, derive coefficients a and b via log-log linear regression, and calculate (10**b)*(x**a).
    Args:
        x (_float|np.ndarray_): x
        y (_float|np.ndarray_): y
        mask (_boolean_)    
    """
    xs = x[mask]
    ys = y[mask]
    a, b = np.polyfit(np.log10(xs), np.log10(ys), 1)
    return partial(func, a=a, b=b)

def create_funcs_to_connect(x, y, bounds):
    """return loglog_interp and loglog_polyfit funcs

    Args:
        bounds (_list_): like a [(min1, max1), (min2, max2)]
    """
    f_mid = loglog_interp(x, y)  # 中央の補間

    (low_min, low_max), (high_min, high_max) = bounds

    f_low  = loglog_power_fit(x, y, (x >= low_min)  & (x < low_max))
    f_high = loglog_power_fit(x, y, (x >= high_min) & (x < high_max))

    return [f_low, f_mid, f_high]

class SpreadWidth:
    def __init__(self):
        
        lateral_sample = pd.read_csv(
            os.path.join(package_path, "data", "normalized_lateral_spreadwidth.csv"),
            index_col=0,
        )
        vertical_sample = pd.read_csv(
            os.path.join(package_path, "data", "normalized_vertical_spreadwidth.csv"),
            index_col=0,
        )
        # super().__init__(windspeed, wether, stab_class)
        # print(f"atmosphere stability classfication : {self.stability_class}")
        # print(f"pasquill-gifford chart approx (y):\n {self.pas_giff_y}")
        # print(f"pasquill-gifford chart approx (z):\n {self.pas_giff_z}")
        self.func_lateral = {}
        self.func_vertical = {}
        self.interval_lateral = {}
        self.interval_vertical = {}
        backline = {
            "A": 1e3,
            "B": 2e3,
            "C": 1e4,
            "D": 5e4,
            "E": 5e4,
            "F": 5e4,
        }

        for lb in classification_label:
            # ---- lateral ----
            y = lateral_sample[lb].dropna()
            x = lateral_sample.loc[y.index, "dist"]

            self.interval_lateral[lb] = [
                (x.min(), 1e3),
                (1e4, x.max())
            ]
            self.func_lateral[lb] = create_funcs_to_connect(x, y, self.interval_lateral[lb])

            # ---- vertical ----
            y = vertical_sample[lb].dropna()
            x = vertical_sample.loc[y.index, "dist"]

            self.interval_vertical[lb] = [
                (x.min(), 2e2),
                (backline[lb], x.max())
            ]
            self.func_vertical[lb] = create_funcs_to_connect(x, y, self.interval_vertical[lb])



    def lateral(self, x, lb):
        xx = np.asarray(x)
        y = connect_smoothly_multi(
            xx, 
            self.func_lateral[lb], 
            self.interval_lateral[lb], 
            transform=np.log10)
        return y.item() if np.isscalar(x) else y

    def vertical(self, x, lb):
        xx = np.asarray(x)
        z = connect_smoothly_multi(
            xx, 
            self.func_vertical[lb], 
            self.interval_vertical[lb], 
            transform=np.log10
        )
        return z.item() if np.isscalar(x) else z
