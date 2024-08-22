import numpy as np

from rsr import sensor_bandpass


def main():
    # Data from NASA Ocean Color
    # https://oceancolor.gsfc.nasa.gov/resources/docs/rsr_tables/
    f0_hs = np.loadtxt("f0.txt", skiprows=15)
    rayleigh_bodhaine_hs = np.loadtxt("rayleigh_bodhaine.txt", skiprows=16)
    k_no2_hs = np.loadtxt("k_no2.txt", skiprows=19)
    k_o3_anderson_hs = np.loadtxt("k_o3_anderson.txt", skiprows=19)

    sensor = "goci2"
    sensor_RSR = "./RSR/gk2b_goci2_RSR.h5"
    sensor_bandpass(
        sensor, sensor_RSR, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs, f0_hs
    )

    sensor = "himawari_8_ahi"
    sensor_RSR = "./RSR/himawari_8_ahi_RSR.h5"
    sensor_bandpass(
        sensor, sensor_RSR, rayleigh_bodhaine_hs, k_o3_anderson_hs, k_no2_hs, f0_hs
    )


if __name__ == "__main__":
    main()
