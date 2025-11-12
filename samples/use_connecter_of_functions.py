import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# プロジェクトルートを sys.path に追加
# __file__ = samples/a.py
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from diffusion_model.func import connect_smoothly_multi, weight

x = np.linspace(0, 2*np.pi*3, 1000)
func1 = lambda x:2*x
func2 = lambda x:4*x
func3 = lambda x:2*np.sin(x)
funcs=[
  func1, 
  func2, 
  func3 
]
intervals = [
  (np.pi, 2*np.pi), 
  (5*np.pi, 6*np.pi)
]
y = connect_smoothly_multi(x, funcs=funcs, mix_intervals=intervals)

plt.figure()
plt.plot(x, y)
plt.show()
