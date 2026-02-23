from pathlib import Path

# PROJ_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
# OUT_DIR = BASE_DIR / "samples" / "sim_result"
OUT_DIR = Path(r"C:\Users\symph\workspace\2026_thesis\fig")

MPL_STYLE = "THESIS"
MPL_STYLES = {
    "THESIS": "forThesis.mplstyle",
    "ABSTRACT": "forAbst.mplstyle",
    "PRESENTATION": "forPresen.mplstyle",
}
MPLSTYLE_PATH = BASE_DIR / MPL_STYLES[MPL_STYLE]

EXT="pdf"
