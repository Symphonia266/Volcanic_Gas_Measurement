# coding: utf-8
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gas_simulation.diffusion_model.diffuse_plume import DiffusePlumeLidar as DP
from gas_simulation.model import MeasurementModel as MM

windspeed = 2
stab_class = "A"
model = MM(windspeed=windspeed, wind_direction=45, elevation=0, stab_class=stab_class)
model.entry_source(radius=5, cnt=(50, -20), He=2)
model.entry_gases("SO2", 2.5 * 1e4, 0.0)
model.entry_gases("H2S", 1.0 * 1e4, 0.0)
model.entry_gases("O3", 0.0, 0.005)
model.show_gases()

model.diffusemodel.clear_source()
model.entry_source(radius=5, cnt=(50, -40), He=2)
model.update_diffuse(windspeed=windspeed, wind_direction=90, stab_class=stab_class)
model.show_gases()
input("ENTER ANY KEY......")

tau1, tau2 = model.transmittance("N2", "O2", False, False)
# ...existing code...

# --- transmittance の簡易テスト（smoke test）---
import inspect

print("\n--- transmittance smoke test ---")
if hasattr(model, "transmittance"):
    try:
        sig = inspect.signature(model.transmittance)
        print("transmittance signature:", sig)

        # まずよくある呼び出し方 (scat1, scat2, dir1, dir2) を試す
        try:
            res = model.transmittance("O2", "N2", False, True)
            # 返り値がタプルなら各要素の shape を表示
            if isinstance(res, tuple):
                print("transmittance returned tuple of length", len(res))
                for i, r in enumerate(res):
                    try:
                        print(f"  [{i}] shape: {np.asarray(r).shape}")
                    except Exception:
                        print(f"  [{i}] type:", type(r))
            else:
                print(
                    "transmittance returned:",
                    type(res),
                    "shape:",
                    np.asarray(res).shape,
                )
        except Exception as e1:
            print("call (scat1,scat2,dir1,dir2) failed:", repr(e1))

            # フォールバック: ranges と wl スタイルで試す
            ranges = np.linspace(1.0, 100.0, 50)
            wl_try = None
            if hasattr(model, "wl"):
                # model.wl が dict なら laser を使う
                try:
                    wl_arr = np.atleast_1d(
                        model.wl.get("laser", list(model.wl.values())[0])
                    )
                    # 適当に代表波長を抽出（長さが大きければ間引く）
                    wl_try = float(wl_arr[len(wl_arr) // 2])
                except Exception:
                    wl_try = None

            if wl_try is not None:
                try:
                    res2 = model.transmittance(ranges, wl_try)
                    print(
                        "transmittance(ranges, wl) succeeded; return type:",
                        type(res2),
                        "shape:",
                        np.asarray(res2).shape,
                    )
                except Exception as e2:
                    print("fallback call (ranges,wl) failed:", repr(e2))
            else:
                print("no suitable wl found for fallback call")
    except Exception as e:
        print("introspection/call failed:", repr(e))
else:
    print("model has no attribute 'transmittance'")

# ...existing code...
