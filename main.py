import numpy as np

from rsr import sensor_bandpass, load_spectral_source_data


def main():

    f0_hs, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs = (
        load_spectral_source_data()
    )
    sensor_RSR = "Spectral_Response_Function/oceancolor_format/gf6_pms_RSR.nc"
    sensor_bandpass(sensor_RSR, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs, f0_hs)


if __name__ == "__main__":
    main()
