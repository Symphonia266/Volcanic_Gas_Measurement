import os
import pandas as pd 

def to_rc_dict(dict):
    return {f"{k1}.{k2}": v for k1, d in dict.items() for k2, v in d.items()}

def export(dir: str, name: str, x: dict | pd.DataFrame):
    filename = dir + name
    # ディレクトリが存在しなければ生成
    os.makedirs(dir, exist_ok=True)

    # 　ファイルが存在すれば上書きモード（mode="a"）で書き込み
    if not os.path.isfile(filename):
        with pd.ExcelWriter(filename) as writer:
            if type(x) == dict:
                for key in x.keys():
                    x[key].to_excel(writer, sheet_name=key, index=True)
            else:
                x.to_excel(writer, index=True)
    else:
        with pd.ExcelWriter(filename, mode="a", if_sheet_exists="replace") as writer:
            if type(x) == dict:
                for key in x.keys():
                    x[key].to_excel(writer, sheet_name=key, index=True)
            else:
                x.to_excel(writer, index=True)
