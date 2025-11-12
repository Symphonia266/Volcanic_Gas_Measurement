from operator import imod
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, field

@dataclass
class LidarModel:
    E0:float
    A:float
    M: float
    eta: float
    q: float
    dR: float 
    Bj: float
    F: float
    D: float
    alt: float

    self.

    def power(self, ):
        return C*beta*tau
      

class AtmosphereModel:
    gas_profiles: dict = field(default_factory=lambda: {"CO2": None, "H2O": None})
    
    def set_profile(self, gas_name: str, profile: np.ndarray):
        """ガス濃度分布を設定"""
        self.gas_profiles[gas_name] = profile
    
    def get_profile(self, gas_name: str):
        return self.gas_profiles.get(gas_name)

class MeasurementModel:
  def __init__(
      lidar_param, 
      obs_y
  ) 
      self.

1. 大気モデル（気体濃度分布など）を設定する
2. ライダーモデル（ショットエネルギーなど）を設定する
3. シミュレーションレンジ（距離rや波長wl）を設定する
4. 大気モデル・ライダーモデルから受信信号強度を導出する
5. 受信信号強度から推定濃度分布を導出する
6. 推定濃度導出分布と設定した大気モデルを比較して精度を確認する
