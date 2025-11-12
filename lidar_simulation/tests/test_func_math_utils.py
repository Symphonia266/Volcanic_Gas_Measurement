import .func.math_utils

if __name__ == "__main__":
    # print("this module with my_sim")
    # from matplotlib import pyplot as plt
    # x = np.linspace(-2, 2, 100)
    # mean = 0

    # # std = 0.5
    # # FWHM = 2*np.power(2*np.log(2), 1/2) * std
    # # y = gaus(x  , std, mean)
    # # b = gaus(std, std, mean)

    # FWHM = 1
    # std = FWHM / 2 * np.power(2 * np.log(2), 1 / 2)
    # y = gaus(x, std, mean, FWHM=FWHM)
    # b = gaus(FWHM / 2, std, mean, FWHM=FWHM)

    # print("x   : {:.1f} - {:.1f} / {:.1f} [nm]".format(x[0], x[1], x[1] - x[0]))
    # print("FWHM: {:.2f} [nm]".format(FWHM))
    # print(r"±\sigma  :{:.2f} [nm]".format(std))

    # fig, ax = plt.subplots(1, 1)
    # ax.grid()
    # ax.plot(x, y)

    # # ax.annotate(r" 2$\sigma$={:.2f} [nm]".format(std*2), xy=[-std, b], xytext=[std, b], va='center',
    # #           arrowprops=dict(arrowstyle='<|-|>',
    # #                           # connectionstyle='arc3',
    # #                           # facecolor='C0',
    # #                           # edgecolor='C0'
    # #                           )
    # # )
    # # ax.vlines( std, ymin=0, ymax=b, ls="--", color="dimgrey")
    # # ax.vlines(-std, ymin=0, ymax=b, ls="--", color="dimgrey")

    # ax.annotate(
    #     " FWHM={:.2f} [nm]".format(FWHM),
    #     xy=[-FWHM / 2, b],
    #     xytext=[FWHM / 2, b],
    #     va="center",
    #     arrowprops=dict(
    #         arrowstyle="<|-|>", connectionstyle="arc3", facecolor="C0", edgecolor="C0"
    #     ),
    # )
    # ax.vlines(FWHM / 2, ymin=0, ymax=b, ls="--", color="dimgrey")
    # ax.vlines(-FWHM / 2, ymin=0, ymax=b, ls="--", color="dimgrey")

    # # ax.vlines(mean-1, ymin=0, ymax=y[x==mean-1], ls="--")
    # # ax.vlines(mean+1, ymin=0, ymax=y[x==mean+1], ls="--")
    # ax.set_xlim(min(x), max(x))
    # ax.set_ylim(0, None)
    # ax.set_xlabel("wavelength [nm]")
    # ax.set_ylabel("intensity")
    # plt.show(block=False)
    # input()
