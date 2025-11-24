#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

# =============================
# Constants
# =============================
IGNORE_DIRS = {".git", "__pycache__"}
IGNORE_FILES = {"__init__.py", "README.md"}

DATA_EXT = {".csv", ".json", ".xlsx"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".gif"}

# Emoji
EMOJI_FILE = "📄"
EMOJI_PY = "🐍"
EMOJI_DIR = "📁"
EMOJI_PACKAGE = "📦"
EMOJI_DATA = "📊"
EMOJI_IMG = "🖼"
EMOJI_OMIT = "~~~"

# Colors
RESET = "\033[0m"
WHITE, GREEN, BLUE, CYAN, MAGENTA = "\033[97m", "\033[92m", "\033[94m", "\033[96m", "\033[95m"

FILE_COLORS = {
    ".py": WHITE,
    **dict.fromkeys(DATA_EXT, GREEN),
    **dict.fromkeys(IMG_EXT, CYAN)
}

# =============================
# Utility functions
# =============================
def is_package_dir(path: Path) -> bool:
    return (path / "__init__.py").exists()

def colorize(name: str, no_color: bool) -> str:
    if no_color: return name
    return f"{FILE_COLORS.get(Path(name).suffix.lower(), RESET)}{name}{RESET}"

def get_icon(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py": return EMOJI_PY
    if ext in IMG_EXT: return EMOJI_IMG
    if ext in DATA_EXT: return EMOJI_DATA
    return EMOJI_FILE

def should_hide(entry: Path, args) -> bool:
    if entry.suffix.lower() in DATA_EXT and args.hide_data: return True
    if entry.suffix.lower() in IMG_EXT and args.hide_image: return True
    return False

def hidden_summary(entries, args):
    """Return list of pseudo-files for hidden summaries"""
    result = []
    if args.hide_data and any(e.suffix.lower() in DATA_EXT for e in entries if e.is_file()):
        result.append("__hidden_data__")
    if args.hide_image and any(e.suffix.lower() in IMG_EXT for e in entries if e.is_file()):
        result.append("__hidden_img__")
    return result

# =============================
# Tree scanning
# =============================
def scan_tree(path: Path, prefix: str, args, level: int = 1, max_level: int = None):
    if max_level is not None and level > max_level:
        print(prefix + f"└── {EMOJI_OMIT} (level limit reached)")
        return

    entries = [e for e in path.iterdir() if e.name not in IGNORE_FILES and e.name not in IGNORE_DIRS]

    # Separate directories
    files = [e for e in entries if e.is_file() and not should_hide(e, args)]
    dirs = [e for e in entries if e.is_dir()]
    normal_dirs, package_dirs = sorted([d for d in dirs if not is_package_dir(d)]), sorted([d for d in dirs if is_package_dir(d)])

    # Add hidden summary pseudo-files
    for h in hidden_summary(entries, args):
        files.append(Path(h))

    ordered = files + normal_dirs + package_dirs

    for i, e in enumerate(ordered):
        is_last = i == len(ordered) - 1
        connector, next_prefix = ("└── ", "    ") if is_last else ("├── ", "│   ")

        if e.is_dir() and e.exists():
            icon = EMOJI_PACKAGE if is_package_dir(e) else EMOJI_DIR
            name = e.name if args.no_color else f"{MAGENTA if icon == EMOJI_PACKAGE else BLUE}{e.name}{RESET}"
            print(prefix + connector + f"{icon} {name}/")
            scan_tree(e, prefix + next_prefix, args, level + 1, max_level)
        elif e.name == "__hidden_data__":
            print(prefix + connector + f"{EMOJI_OMIT} {EMOJI_DATA} (data files hidden)")
        elif e.name == "__hidden_img__":
            print(prefix + connector + f"{EMOJI_OMIT} {EMOJI_IMG} (image files hidden)")
        else:
            print(prefix + connector + f"{get_icon(e)} {colorize(e.name, args.no_color)}")

# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(description="📦 Package Tree Viewer")
    parser.add_argument("target", nargs="?", default=".", help="Base directory to display")
    parser.add_argument("--hide-data", action="store_true", help="Hide data files (.csv, .json, .xlsx)")
    parser.add_argument("--hide-image", action="store_true", help="Hide image files (.png, .jpg, .jpeg, .gif)")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--level", "-L", type=int, default=None, help="Maximum tree depth to display")
    parser.add_argument("-H", action="help", help="Show this help message and exit")
    args = parser.parse_args()

    base = Path(args.target).resolve()
    print(base)
    scan_tree(base, "", args, level=1, max_level=args.level)

if __name__ == "__main__":
    main()
