# refer from https://www.jstage.jst.go.jp/article/jriet1972/4/9/4_9_643/_pdf

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from . import package_path, cmap
from ..pasquill_stable_classfication import PGStableClassfication as PG

pas_giff_z = pd.DataFrame(
    [
        ["A", 1.122, 0.0800, 0],
        ["A", 1.514, 0.00855, 1],
        ["A", 2.109, 0.000212, 2],
        ["B", 0.964, 0.1272, 0],
        ["B", 1.094, 0.0570, 1],
        ["C", 0.918, 0.1068, 0],
        ["D", 0.826, 0.1046, 0],
        ["D", 0.632, 0.400, 1],
        ["D", 0.555, 0.811, 2],
        ["E", 0.788, 0.0928, 0],
        ["E", 0.526, 0.370, 1],
        ["E", 0.415, 1.732, 2],
        ["F", 0.784, 0.0621, 0],
        ["F", 0.526, 0.370, 1],
        ["F", 0.323, 2.41, 2],
        ["G", 0.794, 0.0373, 0],
        ["G", 0.637, 0.1105, 1],
        ["G", 0.431, 0.529, 2],
        ["G", 0.277, 3.62, 3],
    ],
    columns=["stability_class", "p1", "p2", "mode"],
)
pas_giff_y = pd.DataFrame(
    [
        ["A", 0.901, 0.426, 0],
        ["A", 0.851, 0.602, 1],
        ["B", 0.914, 0.282, 0],
        ["B", 0.865, 0.396, 1],
        ["C", 0.924, 0.1772, 0],
        ["C", 0.885, 0.232, 1],
        ["D", 0.929, 0.1107, 0],
        ["D", 0.889, 0.1467, 1],
        ["E", 0.921, 0.0864, 0],
        ["E", 0.897, 0.1019, 1],
        ["F", 0.929, 0.0554, 0],
        ["F", 0.889, 0.0733, 1],
        ["G", 0.921, 0.0380, 0],
        ["G", 0.896, 0.0452, 1],
    ],
    columns=["stability_class", "p1", "p2", "mode"],
)


def correct_time(t1, sigma, t2: float = 3):
    return sigma * (t1 / t2) ** 0.2


class SpreadWidth(PG):
    def __init__(self, windspeed, wether, stab_class):
        super().__init__(windspeed, wether, stab_class)
        self.pas_giff_y = (
            pas_giff_y.query(f'stability_class=="{self.stability_class}"')
            .drop("stability_class", axis=1)
            .set_index("mode")
        )
        self.pas_giff_z = (
            pas_giff_z.query(f'stability_class=="{self.stability_class}"')
            .drop("stability_class", axis=1)
            .set_index("mode")
        )
        print(f"atmosphere stability classfication : {self.stability_class}")
        print(f"pasquill-gifford chart approx (y):\n {self.pas_giff_y}")
        print(f"pasquill-gifford chart approx (z):\n {self.pas_giff_z}")

    def lateral(self, x):
        mode = np.zeros_like(x)
        mode[x < 500] = 0
        mode[x >= 500] = 1
        param_1 = self.pas_giff_y.loc[mode.ravel(), "p1"].values.reshape(mode.shape)
        param_2 = self.pas_giff_y.loc[mode.ravel(), "p2"].values.reshape(mode.shape)
        sigma_y = param_1 * x**param_2
        return sigma_y

    def vertical(self, x):
        mode = np.zeros_like(x)
        if self.stability_class == "A":
            mode[x < 300] = 0
            mode[(x >= 300) & (x < 500)] = 1
            mode[x > 500] = 2
        elif self.stability_class == "B":
            mode[x < 500] = 0
            mode[x >= 500] = 1
        elif self.stability_class == "C":
            mode[:] = 0
        elif self.stability_class == "D":
            mode[x < 1000] = 0
            mode[(x >= 1000) & (x < 10000)] = 1
            mode[x > 10000] = 2
        elif self.stability_class == "E":
            mode[x < 1000] = 0
            mode[(x >= 1000) & (x < 10000)] = 1
            mode[x > 10000] = 2
        elif self.stability_class == "F":
            mode[x < 1000] = 0
            mode[(x >= 1000) & (x < 10000)] = 1
            mode[x > 10000] = 2
        else:
            mode[x < 1000] = 0
            mode[(x >= 1000) & (x < 2000)] = 1
            mode[(x >= 2000) & (x < 10000)] = 2
            mode[x > 10000] = 3
        param_1 = self.pas_giff_z.loc[mode.ravel(), "p1"].values.reshape(mode.shape)
        param_2 = self.pas_giff_z.loc[mode.ravel(), "p2"].values.reshape(mode.shape)
        sigma_z = param_1 * x**param_2
        return sigma_z


if __name__ == "__main__":

    # type "py -m diffusion_model.pasquill"
    import matplotlib.pyplot as plt
    from grid_space import grid_3Dspace

    windspeed = 2
    wether = "night"
    Q = 10 * 1e6
    He = 5

    x1 = np.linspace(0, 10000, 100) + 0.001
    x2 = np.linspace(0, 10000, 1000) + 0.001
    y = np.linspace(-50, 50, 51)
    z = np.linspace(0, 40, 21)

    gs1 = grid_3Dspace(x1, y, z)
    gs2 = grid_3Dspace(x2, y, z)
    model = SpreadWidth(windspeed, wether, None)

    def plume(x, y, z):
        C = (
            Q
            * np.exp(-(y**2) / (2 * model.lateral(x) ** 2))
            * (
                np.exp(-((z - He) ** 2) / (2 * model.vertical(x) ** 2))
                + np.exp(-((z + He) ** 2) / (2 * model.vertical(x) ** 2))
            )
            / (2 * np.pi * model.vertical(x) * model.vertical(x) * windspeed)
        )
        return C

    C1 = plume(gs1.X, gs1.Y, gs1.Z)
    C2 = plume(gs2.X, gs2.Y, gs2.Z)

    plt.figure()
    plt.plot(gs1.x_g, C1[gs1.serch(None, 0, He)], label="N=1e2")
    plt.plot(gs2.x_g, C2[gs2.serch(None, 0, He)], label="N=1e3")
    # plt.legend()
    plt.xlabel("distance [m]")
    plt.ylabel("diffuse width [m/3min]")
    plt.show(block=False)
    input()
