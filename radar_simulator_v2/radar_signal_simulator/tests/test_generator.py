import os

import numpy as np
import matplotlib.pyplot as plt

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary
from config.operating_submode import OperatingSubmode
from config.target import Target
from radar_signal.generator import RadarSimulator

def main():
    # print cwd
    cwd = os.getcwd()
    print(cwd)
    radar = RadarConfig.load("../config/configs/radar_config.json")
    submodes = SubmodeLibrary.load("../config/configs/submodes.json")
    op = OperatingSubmode.load("../config/configs/operating_submode.json")
    targets = Target.load_list("../config/configs/targets.json")

    sim = RadarSimulator(radar, submodes)
    signal = sim.generate(targets, op, noise_sigma=0.0)

    nrows, ncols = radar.array.rows, radar.array.cols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True)

    for elem in range(nrows * ncols):
        r = elem // ncols
        c = elem % ncols
        ax = axes[r, c]

        power = np.abs(signal[elem]) ** 2      # shape (nrg, npri)
        power_db = 10 * np.log10(power + 1e-12)
        ax.imshow(
            power_db.T,   # transpose so x=range gate, y=PRI
            origin="lower",
            aspect="auto",
            cmap="viridis"
        )
        ax.set_title(f"Elem {elem}")

    fig.supxlabel("Range gate")
    fig.supylabel("PRI index")
    fig.suptitle("Element Power |signal|^2 (dB)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()