import os

# ANSI colors
WHITE = "\033[97m"
BLUE = "\033[94m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# ignore 対象
IGNORE_DIRS = {".git", "__pycache__"}
IGNORE_FILES = {"__init__.py"}

# ファイル拡張子別の色
DATA_EXTS = {".csv", ".json", ".xlsx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}

def get_color(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".py":
        return WHITE
    elif ext in DATA_EXTS:
        return GREEN
    elif ext in IMAGE_EXTS:
        return MAGENTA
    else:
        return WHITE  # デフォルトは白

def is_package_dir(path):
    """__init__.py を含むフォルダなら True"""
    return os.path.isfile(os.path.join(path, "__init__.py"))

def print_tree(path=".", prefix=""):
    path = os.path.abspath(path)
    entries = [e for e in os.listdir(path) if e not in IGNORE_DIRS and e not in IGNORE_FILES]

    # ファイルとディレクトリに分離
    files = sorted([e for e in entries if os.path.isfile(os.path.join(path, e))])

    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]

    # 🔥 ここで通常ディレクトリ → パッケージディレクトリ の順に並べ替え
    normal_dirs = []
    package_dirs = []

    for d in dirs:
        if is_package_dir(os.path.join(path, d)):
            package_dirs.append(d)
        else:
            normal_dirs.append(d)

    dirs_sorted = sorted(normal_dirs) + sorted(package_dirs)

    # 全体の順番：ファイル → ディレクトリ
    ordered_entries = files + dirs_sorted
    entries_count = len(ordered_entries)

    for i, entry in enumerate(ordered_entries):
        full_path = os.path.join(path, entry)
        connector = "└── " if i == entries_count - 1 else "├── "
        new_prefix = "    " if i == entries_count - 1 else "│   "

        if os.path.isdir(full_path):
            print(prefix + connector + BLUE + entry + "/" + RESET)
            print_tree(full_path, prefix + new_prefix)
        else:
            color = get_color(entry)
            print(prefix + connector + color + entry + RESET)


if __name__ == "__main__":
    print_tree()
