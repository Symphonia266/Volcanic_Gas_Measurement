
import scipy.constants as const

def nu(wl):
    """the frequency[Hz] by a wavelength [nm]

    Args:
        wl (_float_): wavelength[nm]

    Return:
        (_float_): frequency [cm-1]
    """

    return const.c / 100 * 1e9 / wl


def trans_wn_wl(w):
    """wavenumber[cm-1] <-> wavelength[nm]"""
    return 1e7 / w


def wl_shift(wl, sft, dir: bool, *, inv: bool = False):
    """wavelength raman shifter(stokes/anti-stokes)

    Attributes:

        wl (_float_): shifted wavelength [nm]
        sft (_float_): ramanshift wavenumber [cm-1]
        dir (_bool_): stokes (False) / anti-stokes (True)
        inv (_bool_): invert translation, on is True

    Return:
        (_float_): translated wavelength [nm]

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


def overlap(r) -> float:
    """
    overlap coefficient, with Laser and field of view

    Args:

    Returns:
        _int_: _description_
    """
    return 1.0
