import numpy as np
import pandas as pd
from scipy import constants as const
from scipy.interpolate import interp1d

from . import package_path, data_dir, data_file
from .consts import gases


def nu(wl):
    """the frequency[Hz] by a wavelength [nm]

    Args:
        wl (_float_): wavelength[nm]

    Returns:
        float: frequency [cm-1]
    """

    return const.c / 100 * 1e9 / wl


def trans_wn_wl(w):
    """wavenumber[cm-1] <-> wavelength[nm]
    
    Args:
        wl (_float_): wavenumber[cm-1] / wavelength[nm]

    Returns:
        float: [inverse] wavelength[nm] / wavenumber[cm-1]
    """
    
    return 1e7 / w


def wl_shift(wl, sft, dir: bool, *, inv: bool = False):
    """wavelength raman shifter (stokes/anti-stokes)

    Attributes:

        wl (_float_): shifted wavelength [nm]
        sft (_float_): ramanshift wavenumber [cm-1]
        dir (_bool_): stokes (False) / anti-stokes (True)
        inv (_bool_): invert translation, on is True

    Returns:
        float: translated wavelength [nm]

    Example:

    ::

        print("=== ENTER THE WAVELENGTH YOU WANNA TRANSLATION ===", "\\n")
        print("wl_base [nm]: ", end="")
        wl = float(input())
        sft_O2 = main_gases.at["O2", "shift_wn"]
        sft_N2 = main_gases.at["N2", "shift_wn"]

        wl_shift_O2_st = wl_shift(wl, sft_O2, False)
        wl_shift_O2_ast = wl_shift(wl, sft_O2, True)

        wl_shift_N2_st = wl_shift(wl, sft_N2, False)
        wl_shift_N2_ast = wl_shift(wl, sft_N2, True)

        print("\\n", "    a-stokes     *base       stokes")
        print("O2:   {:.2f}  <-  {:.2f}  ->  {:.2f}   [nm]".format(
            wl_shift_O2_ast, wl_base, wl_shift_O2_st
        ))
        print("N2:   {:.2f}  <-  {:.2f}  ->  {:.2f}   [nm]".format(
            wl_shift_N2_ast, wl_base, wl_shift_N2_st
        ))

    """
    sign = -1 if inv else 1
    return trans_wn_wl(trans_wn_wl(wl) + sign * (sft if dir else -sft))

def gen_conbi(wl, scat1, scat2, dir1, dir2):
    wl_s1 = wl_shift(wl, gases.at[scat1, "sft"], dir1)
    wl_s2 = wl_shift(wl, gases.at[scat2, "sft"], dir2)
    return wl_s1, wl_s2


def ratio_ASTK(
    # Temp: float, gas: str, v0: float
) -> float:
    """return anti-stokes intensity ratio with stokes one.

    Args:
        Temp (_float_): the temperature of gases [K].
        gas (_float_): the kind of the gas.
        v0 (_float_): base wavenumber [cm-1]

    Returns:
        float: anti-stokes signal / stokes signal ratio

    Example:

    ::

        temp:float = 273+100 # 100 cel. degree
        wl:np.ndarray = np.linspace(220, 400, 1000) # 220-400 nm laser
        r:float = r_DIAL_volcano_model.ratio_ASTK(temp, "N2", wl)

        plt.figure(layout="tight")
        plt.scatter(wl, r)
        plt.title(r"N$_2$ $\\dfrac{I_{ASTK}}{I_{STK}}$ wavelength feat.")
        plt.xlabel(r"laser wavelength [nm]")
        plt.ylabel(r"the ratio ($\\dfrac{I_{ASTK}}{I_{STK}}$)")
        plt.show()

    """
    # shift: float = main_gases.at[gas, "shift_wn"]
    # t1: np.ndarray = np.power((v0 + shift) / (v0 - shift), 4)
    # t2: np.ndarray = np.exp(-(const.h * shift) / (const.k * Temp))
    # return t1 * t2
    # return np.exp(-const.hbar * shift / const.k / Temp)
    return 0.1

def gaus(
    x,
    std: float | None = None,
    mean: float = 0.0,
    *,
    HWHM: float | None = None,
    FWHM: float | None = None,
    normalize: str = "pdf",
):
    """
    generater gausian function

    Args:
        x (_type_): _description_
        std (_type_): _description_
        mean (int, optional): _description_. Defaults to 0.
        HWHM (_type_, optional): _description_. Defaults to None.
        FWHM (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_

    Example:
    ::

        x = np.linspace(-2, 2, 100)
        mean = 0

        # std = 0.5
        # FWHM = 2*np.power(2*np.log(2), 1/2) * std
        # y = gaus(x  , std, mean)
        # b = gaus(std, std, mean)

        FWHM = 1
        std = FWHM / 2 * np.power(2 * np.log(2), 1 / 2)
        y = gaus(x, std, mean, FWHM=FWHM)
        b = gaus(FWHM / 2, std, mean, FWHM=FWHM)

        print("x   : {:.1f} - {:.1f} / {:.1f} [nm]".format(x[0], x[1], x[1] - x[0]))
        print("FWHM: {:.2f} [nm]".format(FWHM))
        print(r"±\\sigma  :{:.2f} [nm]".format(std))

        fig, ax = plt.subplots(1, 1)
        ax.grid()
        ax.plot(x, y)

        # ax.annotate(r" 2$\\sigma$={:.2f} [nm]".format(std*2), xy=[-std, b], xytext=[std, b], va='center',
        #           arrowprops=dict(arrowstyle='<|-|>',
        #                           # connectionstyle='arc3',
        #                           # facecolor='C0',
        #                           # edgecolor='C0'
        #                           )
        # )
        # ax.vlines( std, ymin=0, ymax=b, ls="--", color="dimgrey")
        # ax.vlines(-std, ymin=0, ymax=b, ls="--", color="dimgrey")

        ax.annotate(
            " FWHM={:.2f} [nm]".format(FWHM),
            xy=[-FWHM / 2, b],
            xytext=[FWHM / 2, b],
            va="center",
            arrowprops=dict(
                arrowstyle="<|-|>", connectionstyle="arc3", facecolor="C0", edgecolor="C0"
            ),
        )
        ax.vlines(FWHM / 2, ymin=0, ymax=b, ls="--", color="dimgrey")
        ax.vlines(-FWHM / 2, ymin=0, ymax=b, ls="--", color="dimgrey")

        # ax.vlines(mean-1, ymin=0, ymax=y[x==mean-1], ls="--")
        # ax.vlines(mean+1, ymin=0, ymax=y[x==mean+1], ls="--")
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(0, None)
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("intensity")
        plt.show(block=False)
        input()

    """
    x_arr = np.asarray(x, dtype=float)

    if std is None:
        if FWHM is not None:
            std_val = float(FWHM) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        elif HWHM is not None:
            std_val = float(HWHM) / np.sqrt(2.0 * np.log(2.0))
        else:
            raise ValueError('gaus: specify either "std", "FWHM" or "HWHM"')
    else:
        std_val = float(std)

    if not (np.isfinite(std_val) and std_val > 0.0):
        raise ValueError('gaus: "std" must be a positive finite value.')

    z = (x_arr - mean) / std_val
    out = np.exp(-0.5 * z * z) 
    if normalize == "pdf":
        out  /= (std_val * np.sqrt(2.0 * np.pi))
    elif normalize == "peak":
        out = out
    else:
        raise ValueError('gaus: "normalize" must be "pdf" or "peak"')

    return out

def effective(
    x, 
    f, 
    *, 
    spec_sideband=2,
    N=100,
    gaus_kwargs={"mean":0, "FWHM":1, "normalize":"peak"},
    **func_kwargs,
):
    """effective _summary_

    Parameters
    ----------
    x : _float_
        _description_
    f : _function_
        _description_
    spec_sideband : _int_
        (optional) _description_, by default 2
    N : _int_
        (optional) _description_, by default 100
    gaus_kwargs : _dict_
        (optional) _description_, by default {"mean":0, "FWHM":1, "normalize":"peak"}

    Returns
    -------
    _type_
        _description_
    """    
    gaus_kwargs = dict(gaus_kwargs or {})
    
    x = np.asarray(x)
    # 1D の「波長軸」を最後に持ってくる（ただし通常はそのまま）
    *leading, Nwl = x.shape if x.ndim >= 1 else ([], 1)

    spec = np.linspace(-spec_sideband, spec_sideband, N)
    
    weight = gaus(spec, **gaus_kwargs)
    weight = weight/weight.sum()
    
    xx = x[..., :, np.newaxis] + spec[np.newaxis, :]

    yy = f(xx, **func_kwargs)

    y = (weight[np.newaxis, :]*yy).sum(axis=-1)
    return y

def effective_from_interp(
    interp_func, 
    *, 
    spec_sideband=2, 
    N=100, 
    gaus_kwargs={"mean":0, "FWHM":1, "normalize":"peak"},
):
    """
    interp1d 等の波長->断面積関数を受け取り、
    指定したスペクトル幅での有効断面積を返す関数を返す。

    返される関数は scalar または 1D array の wl を受け取り
    calc_of_effective を内部で呼んで値を返します。
    """
    def new_func(x):
        return effective(
            x, 
            interp_func, 
            spec_sideband=spec_sideband, 
            N=N, 
            gaus_kwargs=gaus_kwargs or {}, 
        )
    return new_func

def _read_excel_safe(fname, **kwargs):
    p = data_dir / fname
    if not p.exists():
        raise FileNotFoundError(f"file are not found: {p}")
    return pd.read_excel(p, **kwargs)

def load_cross_section(
        fname, 
        *, 
        skiprows=5, 
        wl_col=0, 
        xs_col=1, 
        interp_kwargs=None, 
        effective: bool = False,
        **read_kwargs):
    """
    Excel ファイルを読み、WL/XS 列を取り出して interp1d を返す。
    - skiprows: ヘッダ等のスキップ行数（既定値は 5）
    - interp_kwargs: interp1d に渡す dict（例: {'bounds_error':False, 'fill_value':'extrapolate'}）
    """
    interp_kwargs = dict(interp_kwargs or {})
    df = _read_excel_safe(
        fname, 
        skiprows=skiprows, 
        header=None, 
        usecols=[wl_col, xs_col], 
        names=["WL", "XS"], 
        **read_kwargs
    )
    df = df.dropna(subset=["WL", "XS"]).sort_values("WL")
    wl = df["WL"].to_numpy()
    xs = df["XS"].to_numpy()

    interp = interp1d(wl, xs, **interp_kwargs)
    if  effective:
        return effective_from_interp(interp)

    return interp

def load_cross_section_dict(mapping, *, interp_kwargs=None, **read_kwargs):
    return {
        k: load_cross_section(v, interp_kwargs=interp_kwargs, **read_kwargs) 
        for k, v in mapping.items()
    }
