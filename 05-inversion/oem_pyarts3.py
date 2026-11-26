"""Forward model for the Exercise 5 OEM retrieval (pyarts3)."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarts3 as pa


INPUT_DIR = Path(__file__).parent / "input"
SENSOR_ALTITUDE = 10e3
LINE_CUTOFF = 750e9
H = 6.62607015e-34
C = 299792458.0
K_B = 1.380649e-23


def load_profile(filename=INPUT_DIR / "x_apriori.xml"):
    """Load the legacy ARTS profile into simple NumPy arrays.

    The pyarts2 GriddedField4 stores pressure as its first grid and altitude,
    temperature, and H2O as the three data books.
    """
    field = ET.parse(filename).getroot().find("GriddedField4")
    if field is None:
        raise ValueError(f"No GriddedField4 found in {filename}")

    grids = [node for node in field.findall("Vector") if int(node.get("nelem", "0"))]
    tensors = field.findall("Tensor4")
    if len(grids) != 1 or len(tensors) != 1:
        raise ValueError(f"Unexpected atmospheric profile layout in {filename}")

    pressure = np.fromstring(grids[0].text or "", sep=" ")
    tensor = tensors[0]
    nlevels = int(tensor.get("npages", "0"))
    nbooks = int(tensor.get("nbooks", "0"))
    values = np.fromstring(tensor.text or "", sep=" ")
    if nbooks != 3 or nlevels != pressure.size or values.size != nbooks * nlevels:
        raise ValueError(f"Unexpected profile dimensions in {filename}")

    altitude, temperature, h2o = values.reshape(nbooks, nlevels)
    if not (np.all(np.diff(altitude) > 0) and np.all(np.diff(pressure) < 0)):
        raise ValueError("Expected altitude-ascending, pressure-descending profile")

    return {
        "pressure": pressure,
        "altitude": altitude,
        "temperature": temperature,
        "h2o": h2o,
    }


def _brightness_temperature(freq, intensity):
    a = 2.0 * H * freq**3 / C**2
    x = H * freq / K_B
    return x / np.log1p(a / intensity)


def _dbtdi(freq, intensity):
    a = 2.0 * H * freq**3 / C**2
    x = H * freq / K_B
    log_term = np.log1p(a / intensity)
    return (x * a / (intensity**2 * (1.0 + a / intensity))) / log_term**2


def forward_model(f_grid, atm_profile, retrieval_quantity="H2O", verbosity=0):
    """Calculate upward Planck brightness temperature and its Jacobian.

    Parameters
    ----------
    f_grid : array-like
        Increasing frequency grid in Hz.
    atm_profile : dict
        Profile returned by :func:`load_profile`, with arrays in retrieval
        state order.
    retrieval_quantity : {"H2O", "Temperature"}
        Atmospheric quantity for the Jacobian.
    verbosity : int
        Retained for compatibility with the legacy exercise; unused in pyarts3.

    Returns
    -------
    brightness_temperature : ndarray, shape (n_frequency,)
    jacobian : ndarray, shape (n_frequency, n_profile_levels)
        Derivative with respect to absolute H2O VMR or temperature.
    """
    del verbosity
    if retrieval_quantity not in {"H2O", "Temperature"}:
        raise ValueError("retrieval_quantity must be 'H2O' or 'Temperature'")

    freq = np.asarray(f_grid, dtype=float)
    if freq.ndim != 1 or freq.size < 2 or not np.all(np.diff(freq) > 0):
        raise ValueError("f_grid must be a one-dimensional increasing grid")

    pa.data.download()
    ws = pa.Workspace()
    ws.freq_grid = pa.arts.AscendingGrid(freq)
    ws.abs_speciesSet(species=["H2O", "O2", "N2"])
    ws.ReadCatalogData()

    # Keep catalog lines within the legacy 750 GHz line-wing cutoff of the
    # measurement band, including the nearby O2 band that shapes its baseline.
    ws.abs_bandsSelectFrequencyByLine(
        fmin=max(0.0, float(freq[0]) - LINE_CUTOFF),
        fmax=float(freq[-1]) + LINE_CUTOFF,
    )
    ws.WignerInit()
    ws.spectral_propmat_agendaAuto()

    points = pa.arts.ArrayOfAtmPoint()
    for p, t, q in zip(
        atm_profile["pressure"], atm_profile["temperature"], atm_profile["h2o"]
    ):
        point = pa.arts.AtmPoint()
        point.pressure = float(p)
        point.temperature = float(t)
        point.wind = np.zeros(3)
        point["H2O"] = float(q)
        point["O2"] = 0.21
        point["N2"] = 0.78
        points.append(point)
    ws.atm_fieldFromProfile(
        atm_profile=points,
        alt_grid=pa.arts.AscendingGrid(atm_profile["altitude"]),
    )
    # The lowest state level is below the sensor, but interpolation near the
    # sensor still gives that boundary level a small Jacobian contribution.
    ws.surf_fieldPlanet(option="Earth")
    ws.spectral_rad_space_agendaSet(option="UniformCosmicBackground")
    ws.spectral_rad_surface_agendaSet(option="Blackbody")

    ws.measurement_sensor = []
    ws.jac_targetsInit()
    if retrieval_quantity == "H2O":
        ws.jac_targetsAddSpeciesVMR(species="H2O")
    else:
        ws.jac_targetsAddTemperature()
    ws.jac_targetsFinalize()

    # as_observer makes the sensor position/LOS the end of the path; the
    # upward-looking ray therefore receives cosmic background from space.
    ws.ray_pathGeometric(
        pos=[SENSOR_ALTITUDE, 0.0, 0.0],
        los=[0.0, 0.0],
        as_observer=1,
        max_stepsize=1000.0,
    )
    ws.spectral_radClearskyEmission()

    intensity = np.asarray(ws.spectral_rad)[:, 0]
    bt = _brightness_temperature(freq, intensity)
    jac_intensity = np.asarray(ws.spectral_rad_jac)
    if jac_intensity.ndim != 3 or jac_intensity.shape[1:] != (freq.size, 4):
        raise RuntimeError(
            "Unexpected spectral_rad_jac shape from pyarts3: "
            f"{jac_intensity.shape}"
        )
    jacobian = (jac_intensity[:, :, 0] * _dbtdi(freq, intensity)).T
    if jacobian.shape != (freq.size, len(atm_profile["altitude"])):
        raise RuntimeError(f"Unexpected Jacobian shape: {jacobian.shape}")

    return bt, jacobian


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile = load_profile()
    grid = np.linspace(22.15e9, 22.30e9, 2000)
    bt, jacobian = forward_model(grid, profile)
    if not np.all(np.isfinite(bt)) or not np.all(np.isfinite(jacobian)):
        raise RuntimeError("Forward model returned non-finite values")
    print(
        f"Brightness temperature: {bt.min():.3f} .. {bt.max():.3f} K; "
        f"Jacobian shape: {jacobian.shape}"
    )
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(grid / 1e9, bt)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Brightness temperature [K]")
    fig.tight_layout()
    fig.savefig(plots_dir / "forward_spectrum.pdf")
    plt.close(fig)
