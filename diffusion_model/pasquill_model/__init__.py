import os
import re

# パッケージの絶対パスを取得
package_path = ''.join(re.split(r'(\\)', __file__)[:-1])
cmap = {
    "A": "#FF4B00",
    "B": "#005AFF",
    "C": "#03AF7A",
    "D": "#4DC4FF",
    "E": "#F6AA00",
    "F": "#000000",
}

# 他モジュールから import 可能にする
__all__ = ["package_path"]
