import numpy as np
from scipy import interpolate
import h5netcdf
from pathlib import Path
import pandas as pd


def generate_sensor_SRF(sensor, wavelength, data):
    output_path = Path("Spectral_Response_Function/oceancolor_format")

    bands = np.sum(wavelength[np.newaxis, :] * data, axis=1) / np.sum(data, axis=1)
    bands = np.round(bands, 0).astype(int)
    with h5netcdf.File(output_path.joinpath(f"{sensor}_RSR.nc"), "w") as f:
        dbands = f.create_variable("bands", data=bands, dimensions=("x",))
        dbands.attrs["long_name"] = "bands"

        dbands.attrs["units"] = "nm"
        dbands.attrs["valid_max"] = 3000
        dbands.attrs["vaild_min"] = 100

        dwavelength = f.create_variable(
            "wavelength", data=wavelength, dimensions=("wavelengths",)
        )
        dwavelength.attrs["long_name"] = "wavelength"
        dwavelength.attrs["units"] = "nm"
        dwavelength.attrs["valid_max"] = 3000
        dwavelength.attrs["vaild_min"] = 100

        dRSR = f.create_variable("RSR", data=data, dimensions=("x", "wavelengths"))
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
            f"Spectral_Response_Function/raw_data/himawari_8_ahi_srf/rtcoef_himawari_8_ahi_srf_ch0{i+1}.txt",
            skiprows=4,
        )
        rsr[:, 0] = 1 / rsr[:, 0] * 1e7

        # 插值成1nm
        f = interpolate.interp1d(
            rsr[:, 0], rsr[:, 1], kind="linear", bounds_error=False, fill_value=0
        )
        print(rsr[:, 0].min(), rsr[:, 0].max())
        RSR[i, :] = f(wavelengths)
    return sensor, bands, wavelengths, RSR


def GF_PMS(sensor="gf6"):
    match sensor:
        case "gf6":
            data_name = "GF-6 PMS.xlsx"
        case "gf5":
            data_name = "GF-5 PMS.xlsx"
        case "gf4":
            data_name = "GF-4 PMS.xlsx"
        case _:
            raise ValueError("sensor not supported")

    df = pd.read_excel(f"Spectral_Response_Function/raw_data/{data_name}")

    # bands = ["PAN", "B1", "B2", "B3", "B4"]
    wavelengths = df["波长/nm"].values
    RSR = df[["PAN", "B1", "B2", "B3", "B4"]].values
    return f"{sensor}_pms", wavelengths, RSR.T


def COMS_GOCI():
    sensor = "coms_goci"
    RSR = np.loadtxt(
        "Spectral_Response_Function/raw_data/GOCI_SRF_350_to_1025_normalized.txt",
        skiprows=1,
    )
    wavelengths = RSR[:, 0]
    return sensor, wavelengths, RSR[:, 1:].T


def main():
    # Himawari_8_AHI()

    # sensor, wavelengths, RSR = GF_PMS(sensor="gf4")
    sensor, wavelengths, RSR = COMS_GOCI()
    generate_sensor_SRF(sensor, wavelengths, RSR)


if __name__ == "__main__":
    main()
