# refer from https://www.jstage.jst.go.jp/article/seikatsueisei1957/17/3/17_3_93/_pdf/-char/ja

import numpy as np
from scipy.interpolate import interp1d
from ..pasquill_stable_classfication import (
    classification_table,
    classification_label,
    classify_atomosphere_stability,
)
from ..func import correct_time

lb_h = [0, 10, 25, 30, 45, 60, 75, 90, 105]
lb_n = [0.2, 0.25, 0.33, 0.5]
C_Y_raw = np.array(
    [
        [0.37, 0.21, 0.21, 0.08],
        [0.37, 0.21, 0.12, 0.08],
        [0.21, 0.12, 0.074, 0.074],
        [0.20, 0.11, 0.070, 0.044],
        [0.18, 0.10, 0.062, 0.040],
        [0.17, 0.095, 0.057, 0.037],
        [0.16, 0.086, 0.053, 0.034],
        [0.14, 0.077, 0.045, 0.030],
        [0.12, 0.060, 0.037, 0.034],
    ]
)
C_Z_raw = np.array(
    [
        [0.21, 0.12, 0.74, 0.047],
        [0.21, 0.12, 0.74, 0.047],
        [0.21, 0.12, 0.074, 0.074],
        [0.20, 0.11, 0.070, 0.044],
        [0.18, 0.10, 0.062, 0.040],
        [0.17, 0.095, 0.057, 0.037],
        [0.16, 0.086, 0.053, 0.034],
        [0.14, 0.077, 0.045, 0.030],
        [0.12, 0.060, 0.037, 0.034],
    ]
)


class SpreadWidth:
    def __init__(self):
        self.n = lb_n[x]
        self.C_Y = {}
        self.C_Z = {}
        for lb in classification_label:
            if (self.stability_class == "A") or (self.stability_class == "B"):
                x = 0
            if self.stability_class == "C":
                x = 1
            if self.stability_class == "D":
                x = 2
            else:
                x = 3

            self.C_Y[lb] = interp1d(lb_h, C_Y_raw[:, x])
            self.C_Z[lb] = interp1d(lb_h, C_Z_raw[:, x])

    # lateral, verticalにてloglog線形回帰とax^bの接続スムージングを実装予定
    def lateral(self, x, height, lb):
        return (self.C_Y[lb](height) * x ** (2 - self.n) / 2) ** (1 / 2)

    def vertical(self, x, height):
        return (self.C_Z[lb](height) * x ** (2 - self.n) / 2) ** (1 / 2)


if __name__ == "__main__":
    pass
