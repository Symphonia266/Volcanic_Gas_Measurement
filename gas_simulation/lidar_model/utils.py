import numpy as np
from typing import Self
from dataclasses import dataclass, field, replace

_UNSET = object()


@dataclass
class Coord:
    distance: np.ndarray
    theta_deg: float
    x0: float
    z0: float

    def __post_init__(self):
        theta: float = np.deg2rad(self.theta_deg)
        self.x: np.ndarray = self.x0 + self.distance * np.cos(theta)
        self.z: np.ndarray = self.z0 + self.distance * np.sin(theta)

    def __getitem__(self, key):
        return self.distance[key], self.x[key], self.z[key]

    def with_(
        self,
        *,
        distance: np.ndarray | object = _UNSET,
        theta_deg: float | object = _UNSET,
        x0: float | object = _UNSET,
        z0: float | object = _UNSET,
    ) -> Self:
        return replace(
            self,
            distance=self.distance if distance is _UNSET else distance,
            theta_deg=self.theta_deg if theta_deg is _UNSET else theta_deg,
            x0=self.x0 if x0 is _UNSET else x0,
            z0=self.z0 if z0 is _UNSET else z0,
        )

    def get_xz(self, r):
        theta = np.deg2rad(self.theta_deg)
        x = self.x0 + r * np.cos(theta)
        z = self.z0 + r * np.sin(theta)
        return x, z
