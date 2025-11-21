import numpy as np

def correct_time(t1, spread, t2: float = 3 * 60):
    return spread * (t1 / t2) ** 0.2

def gen_fauntainsource(radius, cnt, N_pt):
    x0, y0= cnt
    x = np.linspace(-radius, radius, N_pt)
    y = np.linspace(-radius, radius, N_pt)
    x, y = np.meshgrid(x, y)

    mask = (x ** 2 + y**2) <= radius**2
    x = x[mask].flatten()
    y = y[mask].flatten()
    q = np.full(x.size, 1/mask.sum())
    return q, x+x0, y+y0
