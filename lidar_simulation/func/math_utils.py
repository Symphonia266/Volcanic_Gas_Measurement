import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..consts import gases

def onoff_swapper(
    flg, 
    arr1: np.ndarray, 
    arr2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    arr_on = np.zeros_like(arr1)
    arr_off = np.zeros_like(arr1)
    arr_on[flg] = arr1[flg]
    arr_on[~flg] = arr2[~flg]
    arr_off[flg] = arr2[flg]
    arr_off[~flg] = arr1[~flg]
    return arr_on, arr_off

def sign(flg):
    return 1 if flg else -1

def except_idx_edge(arr: np.ndarray, bdr: list):
    return ~(
        (arr[:-1] >= bdr[0]) ^ (arr[1:] >= bdr[0])
        | (arr[:-1] < bdr[1]) ^ (arr[1:] < bdr[1])
    )

def gaus(
    x, 
    std, 
    mean: float = 0, 
    *, 
    HWHM=None, 
    FWHM=None
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
    if HWHM:
        std = HWHM / np.power(2 * np.log(2), 1 / 2)
    if FWHM:
        std = FWHM / 2 / np.power(2 * np.log(2), 1 / 2)

    # return np.exp(-(x-mean)**2/(2*std**2))/np.sqrt(2*np.pi*std**2)
    return norm.pdf(x=x, loc=mean, scale=std)

def calc_of_effective(
    x, 
    f, 
    *, 
    spec_sideband=2,
    N=100, 
    FWHM=1,
    args=None


):
    spec = np.linspace(-spec_sideband, spec_sideband, N)
    weight = gaus(spec,None, 0, FWHM=FWHM)
    weight = weight/ np.sum(weight)
    x = np.asarray(x)
    xx = np.array([spec + xi for xi in x])
    if args is None:
        yy = f(xx)
    elif isinstance(args, (tuple, list)):
        yy = f(xx, *args)
    else:
        yy = f(xx, args)    

    y = (weight[np.newaxis, :]*yy).sum(axis=1)
    return y
