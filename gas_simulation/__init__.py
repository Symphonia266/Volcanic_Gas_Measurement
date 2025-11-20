# ...existing code...
from pathlib import Path

# package_path: このパッケージ（lidar_simulation）ディレクトリの Path オブジェクト
package_path = Path(__file__).resolve().parent

# data_dir: package_path 直下の data ディレクトリ
data_dir = package_path / "data"

# 便利関数（任意）: data 内のファイルパスを得る
def data_file(name: str) -> Path:
    return data_dir / name
# ...existing code...
