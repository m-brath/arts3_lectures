"""Generate deterministic synthetic Exercise 5 measurements with pyarts3."""

from pathlib import Path

import numpy as np
import pyarts3 as pa

from oem_pyarts3 import forward_model, load_profile


HERE = Path(__file__).parent
MEASUREMENT_PATH = HERE / "input" / "measurement_arts3.xml"
TRUE_PROFILE_PATH = HERE / "input" / "x_true_arts3.xml"


def _save_gridded_field(path, grid, values, gridname, dataname, comment):
    field = pa.arts.GriddedField1()
    field.grids = [np.asarray(grid, dtype=float)]
    field.gridnames = [gridname]
    field.dataname = dataname
    field.data = np.asarray(values, dtype=float)
    pa.xml.save(field, str(path), comment=comment)


def simulate_measurement():
    profile = load_profile()
    f_grid = np.linspace(22.15e9, 22.30e9, 2000)
    perturbed_h2o = (
        profile["h2o"]
        + 3e-5
        * np.exp(-profile["altitude"] / 50000)
        * 0.5
        * np.sin(2 * np.pi * 0.0001 * profile["altitude"])
        + 0.5e-5
    )
    perturbed_h2o[perturbed_h2o < 0] = profile["h2o"][-1]
    true_profile = {**profile, "h2o": perturbed_h2o}

    brightness_temperature, _ = forward_model(f_grid, true_profile)
    noise_std = 0.05
    seed = 12345
    rng = np.random.default_rng(seed)
    measurement = brightness_temperature + rng.normal(scale=noise_std, size=f_grid.size)

    _save_gridded_field(
        MEASUREMENT_PATH,
        f_grid,
        measurement,
        "Frequency",
        "Brightness temperature",
        f"Simulated with ARTS 3; Gaussian noise std={noise_std} K, seed={seed}.",
    )
    _save_gridded_field(
        TRUE_PROFILE_PATH,
        profile["altitude"],
        perturbed_h2o,
        "Altitude",
        "H2O volume mixing ratio",
        "True H2O profile used to simulate measurement_arts3.xml.",
    )
    print(f"Wrote {MEASUREMENT_PATH}")
    print(f"Wrote {TRUE_PROFILE_PATH}")
    return f_grid, measurement, true_profile


if __name__ == "__main__":
    simulate_measurement()
