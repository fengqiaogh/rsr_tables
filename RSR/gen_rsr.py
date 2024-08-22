import h5py
import numpy as np
from scipy import interpolate


def gen_RSR(sensor, bands, wavelength, data):
    with h5py.File(f"{sensor}_RSR.h5", "w") as f:
        dbands = f.create_dataset(
            "bands",
            data=bands,
        )
        dbands.attrs["long_name"] = "bands"

        dbands.attrs["units"] = "nm"
        dbands.attrs["valid_max"] = 3000
        dbands.attrs["vaild_min"] = 100

        dbands = f.create_dataset(
            "wavelength",
            data=wavelength,
        )
        dbands.attrs["long_name"] = "wavelength"

        dbands.attrs["units"] = "nm"
        dbands.attrs["valid_max"] = 3000
        dbands.attrs["vaild_min"] = 100

        dRSR = f.create_dataset(
            "RSR",
            data=data,
        )
        dRSR.attrs["long_name"] = "Relative Spectral Response"

        dRSR.attrs["units"] = "none"
        dRSR.attrs["valid_max"] = 1.0
        dRSR.attrs["vaild_min"] = 0.0


def Himawari_8_AHI():
    sensor = "himawari_8_ahi"

    # data from https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov13/ir_srf/rtcoef_himawari_8_ahi_srf.html
    bands = np.array([470, 510, 639, 865, 1610, 2256])
    min_wa, max_wa = 400, 2350
    wavelengths = np.arange(min_wa, max_wa, 1)
    RSR = np.zeros((bands.size, wavelengths.size))

    for i in range(len(bands)):

        rsr = np.loadtxt(
            f"RSR/himawari_8_ahi_srf/rtcoef_himawari_8_ahi_srf_ch0{i+1}.txt", skiprows=4
        )
        rsr[:, 0] = 1 / rsr[:, 0] * 1e7

        # 插值成1nm
        f = interpolate.interp1d(
            rsr[:, 0], rsr[:, 1], kind="linear", bounds_error=False, fill_value=0
        )
        print(rsr[:, 0].min(), rsr[:, 0].max())
        RSR[i, :] = f(wavelengths)

    gen_RSR(sensor, bands, wavelengths, RSR)


def main():
    Himawari_8_AHI()


if __name__ == "__main__":
    main()
