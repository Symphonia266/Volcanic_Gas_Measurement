from ..func.optics import wl_shift

if __name__ == "__main__":
    a = wl_shift(341, main_gases.at["N2", "shift_wn"], True, inv=False)
    b = wl_shift(341, main_gases.at["O2", "shift_wn"], True, inv=False)
    c = wl_shift(341, main_gases.at["N2", "shift_wn"], False, inv=False)
    print(a, b)
    print(a, c)
