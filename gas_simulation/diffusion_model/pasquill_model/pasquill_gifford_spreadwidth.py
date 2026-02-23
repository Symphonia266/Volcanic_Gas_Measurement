import os
import numpy as np
import pandas as pd
# from scipy.interpolate import interp1d
from functools import partial
from scipy.interpolate import interp1d 

from . import package_path
from ..pasquill_stable_classfication import (
    # classification_table,
    classification_label,
    # classify_atomosphere_stability,
)
# from ..func import correct_time

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

def extrapolate(x, intercept, slope):
    return (10**intercept) * (x**slope)

def smoothstep(t):
    return 3 * t**2 - 2 * t**3

def weight(x, a, b):
    t = (x - b) / (a - b)
    t = np.clip(t, 0.0, 1.0)
    w = smoothstep(t)
    return w

def connect_smoothly_multi(x, funcs, mix_intervals, transform=None):
    """ 
        If you input func1, func2, func3, func4 
        and the intervals [(a, b), (c, d), (d, e)], the for loop will repeat three times.
        
        When i=0 (left=a, right=b), func1 and func2 are mixed in the interval from a to b, 
        while func2 is calculated as-is in the interval from b to c.
        When i=1 (left=c, right=d), func2 and func3 are mixed in the interval from c to d, 
        while func3 is calculated as-is in the interval from d to d (no mixed interval exists).
        When i=2 (left=c, right=d), func3 and func4 are blended in the interval from d to e. 
        However, since no subsequent blending interval exists, the loop exits without performing the non-blending interval calculation as before.
        
        Outside the for loop, func1 and func4 are simply extrapolated for the mixed intervals at both ends.
    
    Args:
        x (_float|np.ndarray_): The arguments for the function to be connected
        funcs (_list[function, ...]_): The list of functions to be connected
        mix_intervals (_list[tuple, ...]_): The connection interval of functions
        transform (_function_): 
            When an optional function is provided, e.g., np.log10
            the weights are computed on transform(x), transform(left), and transform(right)
    """
    
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    # Outside the first mixing section
    start, _ = mix_intervals[0]
    mask = x < start
    if np.any(mask):
        y[mask] = funcs[0](x[mask])

    # Outside the last mixing section
    _, last_right = mix_intervals[-1]
    mask = x >= last_right
    if np.any(mask):
        y[mask] = funcs[-1](x[mask])
   
    # Mixing sections,  or Non-mixing section in the middle
    for i, (left, right) in enumerate(mix_intervals):

        # mixing section
        mask_mix = (x >= left) & (x < right)
        if np.any(mask_mix):
            if transform is None:
                w = weight(x[mask_mix], left, right)
            else:
                w = weight(transform(x[mask_mix]), transform(left), transform(right))
            y[mask_mix] = w * funcs[i](x[mask_mix]) + (1 - w) * funcs[i + 1](x[mask_mix])

        # non mixising section until next mixing section
        if i < len(mix_intervals) - 1:
            next_left, _ = mix_intervals[i + 1]
            mask_nonmix = (x >= right) & (x < next_left)
            if np.any(mask_nonmix):
                y[mask_nonmix] = funcs[i + 1](x[mask_nonmix])
    return y

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
        return y

    def vertical(self, x, lb):
        xx = np.asarray(x)
        z = connect_smoothly_multi(
            xx, 
            self.func_vertical[lb], 
            self.interval_vertical[lb], 
            transform=np.log10
        )
        return z
