import glob

import h5py
import numpy as np


def gf4_srf(step):
    with h5py.File("gf4_pmi_RSR.h5") as f:
        print(f.keys())
        x = f["wavelength"][:]
        RSR = f["RSR"][:]

    wv = np.arange(250, 4002.5, 2.5)  # 6s sampling

    for band in range(1, 5):
        y = RSR[band, :]
        y_interp = np.interp(wv, x, y, left=0.0, right=0)
        passer = y_interp >= 0.005
        code_block = "%s = ( %d, %0.5f, %0.5f, \n" % (
            "GF4_PMI_%02d" % (band + 1),
            band + step,
            wv[passer][0] / 1000.0,
            wv[passer][-1] / 1000.0,
        )
        splodge = "".join(["%0.5f, " % r for r in y_interp[passer]])
        code_block += "\t\t np.array([%s]))\n\n" % splodge
        print(code_block)
    return band + step


if __name__ == "__main__":
    step = gf4_srf(139)
