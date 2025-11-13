import numpy as np
from scipy.interpolate import interp1d


def correct_time(t1, spread, t2: float = 3 * 60):
    return spread * (t1 / t2) ** 0.2

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
