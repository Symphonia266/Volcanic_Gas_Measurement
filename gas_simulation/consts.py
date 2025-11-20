import numpy as np
import pandas as pd

# s_raman_N2: float = 3.4e-30 *1e-4 # [m2/str.-1] Total
s_raman_N2: float = 2.8e-30 * 1e-4  # [m2/str.-1] Q-branch only
""" Cross-section of absorption for Raman scattering by N2 [m^2] """
# https://www.jstage.jst.go.jp/article/oubutsu1932/42/9/42_9_889/_pdf/-char/ja

r_O2_raman: float = 1.4
""" the xs ratio rO2 for N2"""
s_raman_O2:float = s_raman_N2 * r_O2_raman
""" Cross-section of absorption for Raman scattering by O2 [m^2] """

gases: pd.DataFrame = pd.DataFrame.from_dict(
    {
        "N2": [28.01, 2331, 0.7811, 0.7553, s_raman_N2],
        "O2": [32.00, 1556, 0.2096, 0.2314, 1.4 * s_raman_N2],
        "Ar": [39.94, np.nan, 9.343e-3, 1.28e-2, np.nan],
        "CO2": [
            44.01,
            [1388, 1286],
            3.0e-4,
            4.5e-4,
            [1.2 * s_raman_N2, 0.84 * s_raman_N2],
        ],
        "CO": [28.01, 2145, 1.0e-5, 1.0e-7, 0.9 * s_raman_N2],
        "Ne": [20.18, np.nan, 1.8e-5, 1.2e-7, np.nan],
        "He": [4.00, np.nan, 5.3e-6, 7.3e-7, np.nan],
        "CH4": [
            16.05,
            [2914, 3020],
            1.52e-6,
            8.4e-5,
            [7.5 * s_raman_N2, 1.2 * s_raman_N2],
        ],
        "Kr": [83.7, np.nan, 1.0e-6, 3.0e-6, np.nan],
        "N2O": [44.02, np.nan, 5.0e-7, 8.0e-7, np.nan],
        "H2": [2.02, 4160, 5.0e-7, 3.0e-8, 3.2 * s_raman_N2],
        "O3": [48.0, np.nan, 2.0e-7, 3.0e-8, np.nan],
        "H2O": [18.02, 3651, np.nan, np.nan, 1.6 * s_raman_N2],
        "SO2": [64.06, 1151, np.nan, np.nan, 5.7 * s_raman_N2],
    },
    orient="index",
    columns=["mol-weight", "sft", "vol-ratio", "mass-ratio", "s_ram"],
)
"""
atmospheric gases parameters
---

    **[idx]**  `N2`, `O2`, `Ar`, `CO2`, `CO`, `Ne`, `He`, `CH4`, `Kr`, `N2O`, `H2`, `O3`, `H2O`, `SO2`

    **[col]** `mol-weight`, `shift_wn[cm-1]`, `vol-ratio`, `mass-ratio`, `xs_raman[m2]`
"""
# s_raman_N2: float = 1.64e-29*1e-4 # [m2/sr] ±8% @488.0nm
# ratio_O2_ramanXS_forN2: float = 2.61 # ±5% @488.0nm
# ratio_CO2_ramanXS_forN2: float = 10.6 # ±10% @488.0nm

# generating JIS color pallet

"""
universal color 7 map
---
::

      ---   R   G   B
  [0] ---  255  75   0   鮮やかな黄みの赤   #FF4B00  
  [1] ---    0  90 255   鮮やかな青        #005AFF 
  [2] ---    3 175 122   重厚な青みの緑     #03AF7A 
  [3] ---   77 196 255   シアン色          #4DC4FF 
  [4] ---  246 170   0   鮮やかな黄みの橙   #F6AA00 
  [5] ---  255 241   0   鮮やかな黄        #FFF100 
  [6] ---    0   0   0   漆黒の黒色        #000000 
"""
