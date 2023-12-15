import h5py
import numpy as np
import pandas as pd


def gen_RSR(bands, wavelength, data):
    with h5py.File("gk2b_goci2_RSR.h5", "w") as f:
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


def sensor_bandpass(
    sensor, sensor_RSR, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs, f0_hs
):
    with h5py.File(sensor_RSR) as f:
        band_num = [""]
        nominal_center_wavelength = ["nm"]
        center_wavelength = ["nm"]
        width = ["nm"]
        solar_irradiance = ["W/m^2/um"]
        rayleigh_optical_thickness = [""]
        depolarization_factor = [""]
        k_oz = [""]
        k_no2 = [""]
        for idx, band in enumerate(f["bands"][:]):
            band_num.append(idx + 1)
            nominal_center_wavelength.append(band)
            rot = Spectral_Characterization(
                rayleigh_bodhaine_hs[:, [0, 1]],
                np.array([f["wavelength"][:], f["RSR"][:][idx, :]]).T,
                f0_hs,
            )
            rayleigh_optical_thickness.append(rot)
            k_oz1 = Spectral_Characterization(
                k_o3_anderson_hs,
                np.array([f["wavelength"][:], f["RSR"][:][idx, :]]).T,
                f0_hs,
            )
            k_oz.append(k_oz1)
            k_no21 = Spectral_Characterization(
                k_no2_hs, np.array([f["wavelength"][:], f["RSR"][:][idx, :]]).T, f0_hs
            )
            k_no2.append(k_no21)
            depolar = Spectral_Characterization(
                rayleigh_bodhaine_hs[:, [0, -1]],
                np.array([f["wavelength"][:], f["RSR"][:][idx, :]]).T,
                f0_hs,
            )
            depolarization_factor.append(depolar)
            f0 = Spectral_Characterization(
                f0_hs,
                np.array([f["wavelength"][:], f["RSR"][:][idx, :]]).T,
                np.array([1]),
            )
            solar_irradiance.append(f0 * 10)
        df = pd.DataFrame(
            {
                "Band Num": band_num,
                " Nominal Center Wavelength": nominal_center_wavelength,
                "Solar Irradiance": solar_irradiance,
                "Rayleigh Optical Thickness": rayleigh_optical_thickness,
                "Depolarization Factor": depolarization_factor,
                "k_oz (Ozone)": k_oz,
                "k_no2 (NO2)": k_no2,
            }
        )
        df.to_csv(f"{sensor}_bandpass.csv", index=False)


def Spectral_Characterization(x, rsr, w):
    wa_min = rsr[:, 0].min()
    wa_max = rsr[:, 0].max()
    x_wa_min = np.argwhere(x[:, 0] == wa_min)[0, 0]
    x_wa_max = np.argwhere(x[:, 0] == wa_max)[0, 0]
    if w.size == 1:
        numerator = x[x_wa_min : x_wa_max + 1, -1] * rsr[:, -1]
        denominator = rsr[:, -1]
    else:
        w_wa_min = np.argwhere(w[:, 0] == wa_min)[0, 0]
        w_wa_max = np.argwhere(w[:, 0] == wa_max)[0, 0]

        numerator = (
            x[x_wa_min : x_wa_max + 1, -1] * rsr[:, -1] * w[w_wa_min : w_wa_max + 1, -1]
        )
        denominator = w[w_wa_min : w_wa_max + 1, -1] * rsr[:, -1]

    return numerator.sum() / denominator.sum()
