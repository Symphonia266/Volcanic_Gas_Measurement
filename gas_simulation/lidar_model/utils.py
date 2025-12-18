import numpy as np
from dataclasses import dataclass, field


@dataclass
class Coord:
    distance: np.ndarray
    theta_deg: float
    x0: float
    z0: float

    def __post_init__(self):
        theta = np.deg2rad(self.theta_deg)
        self.x = self.x0 + self.distance * np.cos(theta)
        self.z = self.z0 + self.distance * np.sin(theta)

    def __getitem__(self, key):
        return self.distance[key], self.x[key], self.z[key]

    def get_xz(self, r):
        theta = np.deg2rad(self.theta_deg)
        x = self.x0 + r * np.cos(theta)
        z = self.z0 + r * np.sin(theta)
        return x, z
