import numpy as np
import pandas as pd
import h5netcdf

from pathlib import Path


def load_spectral_source_data(type="1nm"):
    # Data from NASA Ocean Color
    # https://oceancolor.gsfc.nasa.gov/resources/docs/rsr_tables/
    f0_hs = np.loadtxt(f"spectral_source_data/{type}/f0.txt", skiprows=15)
    rayleigh_bodhaine_hs = np.loadtxt(
        f"spectral_source_data/{type}/rayleigh_bodhaine.txt", skiprows=16
    )
    k_no2_hs = np.loadtxt(f"spectral_source_data/{type}/k_no2.txt", skiprows=19)
    k_o3_anderson_hs = np.loadtxt(
        f"spectral_source_data/{type}/k_o3_anderson.txt", skiprows=19
    )
    return f0_hs, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs


def sensor_bandpass(
    sensor_RSR, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs, f0_hs
):
    with h5netcdf.File(sensor_RSR) as f:
        band_num = [""]
        nominal_center_wavelength = ["nm"]
        solar_irradiance = ["W/m^2/um"]
        rayleigh_optical_thickness = [""]
        depolarization_factor = [""]
        k_oz = [""]
        k_no2 = [""]
        for idx, band in enumerate(f["bands"][:]):
            band_num.append(idx + 1)
            wavelength = f["wavelength"][:]
            RSR = f["RSR"][:][idx, :]

            # 中心波长
            nominal_center_wavelength.append(int(band))

            rot = Spectral_Characterization(
                rayleigh_bodhaine_hs[:, [0, 1]],
                np.array([wavelength, RSR]).T,
                f0_hs,
            )
            rot = np.format_float_scientific(rot, precision=3)
            rayleigh_optical_thickness.append(rot)

            k_oz1 = Spectral_Characterization(
                k_o3_anderson_hs,
                np.array([wavelength, RSR]).T,
                f0_hs,
            )
            k_oz1 = np.format_float_scientific(k_oz1, precision=3)
            k_oz.append(k_oz1)

            k_no21 = Spectral_Characterization(
                k_no2_hs, np.array([wavelength, RSR]).T, f0_hs
            )
            k_no21 = np.format_float_scientific(k_no21, precision=3)
            k_no2.append(k_no21)

            depolar = Spectral_Characterization(
                rayleigh_bodhaine_hs[:, [0, -1]],
                np.array([wavelength, RSR]).T,
                f0_hs,
            )
            depolar = np.format_float_scientific(depolar, precision=3)
            depolarization_factor.append(depolar)

            f0 = Spectral_Characterization(
                f0_hs,
                np.array([wavelength, RSR]).T,
                np.array([1]),
            )
            f0 = np.around(f0 * 10, 3)
            solar_irradiance.append(f0)

        df = pd.DataFrame(
            {
                "Band Num": band_num,
                "Nominal Center Wavelength": nominal_center_wavelength,
                "Solar Irradiance": solar_irradiance,
                "Rayleigh Optical Thickness": rayleigh_optical_thickness,
                "Depolarization Factor": depolarization_factor,
                "k_oz (Ozone)": k_oz,
                "k_no2 (NO2)": k_no2,
            }
        )

        sensor = "_".join(Path(sensor_RSR).name.split("_")[:2])
        # df.to_csv(f"bandpass/{sensor}_bandpass.csv", index=False)

        # 将DataFrame转换为Markdown格式
        markdown_output = df.to_markdown(index=False)

        # 将Markdown写入文件
        with open(f"bandpass/{sensor}_bandpass.md", "w") as f_markdown:
            f_markdown.write(markdown_output)

        print(f"Markdown file saved as bandpass/{sensor}_bandpass.md")


def Spectral_Characterization(x, rsr, w=np.array([1])):
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
