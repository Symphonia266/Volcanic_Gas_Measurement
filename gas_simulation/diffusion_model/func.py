import numpy as np

def correct_time(t1, spread, t2: float = 3 * 60):
    return spread * (t1 / t2) ** 0.2

def gen_fauntainsource(radius, cnt,  N_pt):
    x0, y0, z0 = cnt
    x = np.linspace(-radius, radius, N_pt)
    y = np.linspace(-radius, radius, N_pt)
    x, y = np.meshgrid(x, y)
    z = 0

    mask = (x ** 2 + y**2) <= radius**2
    x = x[mask].flatten()
    y = y[mask].flatten()
    q = 1 / mask.sum()
    return x+x0, y+y0, z+z0, q
