# %% Import modules and define functions
"""Calculate and plot clear-sky brightness-temperature Jacobians and opacity
profiles (pyarts3).

This is the ARTS3 migration of the 2.6 exercise 04-jacobian.  It computes:

* the upwelling **brightness temperature** spectrum seen by a satellite at
  800 km looking straight down (nadir),
* the **zenith opacity** spectrum,
* the **altitudinal Jacobian** of the brightness temperature with respect to a
  chosen target (``H2O`` mixing ratio or ``T`` temperature), per atmosphere
  level and per frequency, in K/km,
* the **opacity profile**  $\\tau(z)$  for the selected frequency (running
  integral of the extinction coefficient from the top of the atmosphere down
  to  altitude  ``z``).
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pyarts3 as pa
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# Physical constants (SI)
H = 6.62607015e-34  # Planck constant [J s]
C = 299792458.0  # speed of light [m/s]
K_B = 1.380649e-23  # Boltzmann constant [J / K]

# Profile (midlatitude - summer) with 50 altitudes  0  ->  1.2  m
BNAME = "planets/Earth/afgl/midlatitude-summer/"


def argclosest(array, value):
    """Returns the index in ``array`` which is closest to ``value``."""
    return np.abs(np.asarray(array) - value).argmin()


def tag2tex(tag):
    """Replace all digits in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", str(tag))


def species_from_tags(tags):
    """Translate the lecture species tags to ARTS3 absorption species.

    ARTS 3 uses the ``lines/`` line files in  ``arts - cat - data - trunk``
    for bare element tags (``H2O``, ``O2``, ``N2``), exactly matching the
    ``ReadSpeciesSplitCatalog(basename="lines/")``  bundle that the  2.6
    lecture loads.  The  ``-PWR2022``  /  ``-PWR2021``  /  ``-CKDMT``  suffixed
    tags are separate, much heavier, line sets (in  ``isotopologues/``)
    that are NOT  the  2.6  reference  data.
    """
    allowed = {"H2O", "O2", "N2"}
    arts = []
    for tag in tags:
        base = tag.split("-")[0]
        if base in allowed and base not in arts:
            arts.append(base)
    if not arts:
        raise ValueError(f"No ARTS3 species for input tags {tags}")
    return arts


def _xml_data_root() -> str:
    """Best - effort location of the  ``arts - xml - data`` tree.

    pyarts3 does not expose its data path on the public API, so we fall
    back on the environment or the  standard download  location.
    """
    import os

    candidates = [
        os.getenv("ARTS_DATA_PATH"),
        os.getenv("ARTS_INCLUDE_PATH"),
        os.path.join(os.path.expanduser("~"), ".cache", "arts", "arts-xml-data-trunk"),
        os.path.join(os.path.expanduser("~"), ".cache", "arts"),
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find the arts-xml-data tree. Set ARTS_DATA_PATH. Tried: {candidates}"
    )


def _read_alt_grid(basename: str = BNAME) -> np.ndarray:
    """Read the  50 - point altitude  grid (m) from  the  profile's  ``t.xml``.

    The workspace  object  for  ``atm_field``  does  not  expose the grid; the
    first  ``Vector`` of rank - 1  (the  ``Altitude`` axis)  inside  the
    temperature field holds it.
    """
    path = os.path.join(_xml_data_root(), basename, "t.xml")
    tree = ET.parse(path)
    for node in tree.getroot().iter("Vector"):
        if node.get("rank", "") == "1" and node.get("shape", "") != "1":
            return np.array([float(x) for x in node.text.split()], dtype=float)
    raise ValueError(f"Altitude grid not found in {path}")


def _bt_from_intensity(freq: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Invert the Planck formula:  T =  (h f / kB)  / ln(1 +  2 h f^3 / (c^2 B)).

    Parameters:
        freq: frequency grid [Hz]
        intensity: upwelling spectral intensity [W m^-2 sr^-1 Hz^-1]
    Returns:
        Brightness temperature [K].
    """
    hf_kb = H * freq / K_B
    pref = 2.0 * H * freq**3 / (C**2 * intensity)
    return hf_kb / np.log1p(pref)


def _dbtdI(freq: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Analytic derivative of BT wrt intensity (needed to convert the
    intensity - domain Jacobian into a BT - domain Jacobian).

    dBT/dI =  x  a  / (I^2 ( 1 +  a / I ))  /  ln(1 +  a / I)^2
    where  x =  h f / kB   and   a  =  2 h f^3 / c^2 .
    """
    x = H * freq / K_B
    a = 2.0 * H * freq**3 / C**2
    L = np.log1p(a / I)
    return (x * a / (I**2 * (1.0 + a / I))) / L**2


# -------------------  plotting helpers  -------------------


def plot_brightness_temperature(frequency, y, where=None, ax=None):
    """Plot BT  vs  frequency [GHz] and (optionally) mark a frequency."""
    if ax is None:
        ax = plt.gca()
    ax.plot(frequency / 1e9, y)
    ax.set_xlim(frequency.min() / 1e9, frequency.max() / 1e9)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(r"$T\mathrm{_B}$ [K]")
    if where is not None:
        i = argclosest(frequency, where)
        (l,) = ax.plot(frequency[i] / 1e9, y[i], marker="o", color="tab:red")
        ax.text(
            0.05, 0.9, f"{frequency[i]/1e9:.2f} GHz",
            size="small", color=l.get_color(), transform=ax.transAxes,
        )


def plot_opacity(frequency, opacity, where=None, ax=None):
    """Plot zenith opacity  vs  frequency on log  scale,  optionally mark
    a  frequency."""
    if ax is None:
        ax = plt.gca()
    ax.semilogy(frequency / 1e9, opacity[-1, :])
    ax.set_xlim(frequency.min() / 1e9, frequency.max() / 1e9)
    ax.axhline(1, color="darkgrey", linewidth=0.8, zorder=-1)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Zenith Opacity")
    if where is not None:
        i = argclosest(frequency, where)
        ax.plot(frequency[i] / 1e9, opacity[-1, i], marker="o", color="tab:red")


def plot_jacobian(height, jacobian, jacobian_quantity, ax=None):
    """Plot altitudinal Jacobian  vs  altitude;  label  the  peak."""
    if ax is None:
        ax = plt.gca()
    ax.plot(jacobian, height / 1000.0)
    ax.set_ylim(0.4, 20)
    unit = "K/K/km" if jacobian_quantity == "T" else "K/1/km"
    ax.set_xlabel(f"{tag2tex(jacobian_quantity)} Jacobian [{unit}]")
    ax.set_ylabel("$z$ [km]")
    jac_peak = height[np.abs(jacobian).argmax()] / 1000.0
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    lh = ax.axhline(jac_peak, color="black", zorder=3)
    ax.text(
        1, jac_peak, f"{jac_peak:.2f} km", size="small", ha="right", va="bottom",
        color=lh.get_color(), bbox={"color": "white", "alpha": 0.5},
        zorder=2, transform=trans,
    )


def plot_opacity_profile(height, opacity, ax=None):
    """Plot  $\\tau(z, z_{TOA})$  vs  altitude  on  log-x  scale, label  $\\tau=1$."""
    if ax is None:
        ax = plt.gca()
    ax.semilogx(opacity, height[::-1] / 1000.0)
    ax.set_xlim(1e-8, 1e2)
    ax.set_xticks(10.0 ** np.arange(-8, 4, 2))
    ax.set_xlabel(r"Opacity $\tau(z, z_\mathrm{TOA})$")
    ax.set_ylim(0.4, 20)
    ax.set_ylabel("$z$ [km]")
    try:
        tau1 = height[::-1][np.where(opacity >= 1)[0][0]]
    except IndexError:
        pass
    else:
        tau1 /= 1000
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        lh = ax.axhline(tau1, color="black", zorder=3)
        ax.text(
            0.05, tau1, f"{tau1:.2f} km", va="bottom", size="small",
            color=lh.get_color(), bbox={"color": "white", "alpha": 0.5},
            zorder=2, transform=trans,
        )
        ax.axvline(1, color="darkgrey", linewidth=0.8, zorder=-1)


# -------------------  main API  -------------------


def calc_jacobians(
    jacobian_quantity="H2O",
    species=("N2", "O2", "H2O"),
    fmin=150e9,
    fmax=200e9,
    fnum=200,
    verbosity=0,
):
    """Run a clear-sky radiative-transfer calculation and its Jacobian.

    Parameters:
        jacobian_quantity ("H2O" | "T"):
            Target for the per-atmosphere-level Jacobian.  ``H2O`` uses the
            *log-reliant* (relative VMR) target as in the 2.6 exercise
            (``unit="rel"``);  ``T`` uses additive temperature.
        species (list[str]):  Lecture species tags (N2/O2/H2O).
        fmin, fmax (float):  Frequency range for LBL [Hz].
        fnum (int):  Number of frequency grid points.
        verbosity:  ARTS verbosity level (unused in pyarts3).

    Returns:
        A dict with:
          freq (n_freq,)      [Hz]
          bt   (n_freq,)      BT [K]
          tau  (n_alt, n_freq)  Cumulative opacity  $\\tau(z, TOA)$  [1]  (row 0 = TOA /  ``\\tau=0`` ,  row -1 =  surface /  zenith)
          jac  (n_freq, n_alt)  BT  Jacobian  dBT/dtarget  per  km  [K/km
                                   (relative-VMR)]  (or  [K/K/km]  if  T)
          alt  (n_alt,)       Altitude grid of the 50 atmosphere levels [m]
                              (ascending, surface -> TOA)
    """
    _ = verbosity  # pyarts3 has no per-workspace verbosity
    pa.data.download()
    alt = _read_alt_grid()
    n_alt = len(alt)
    freq = np.linspace(fmin, fmax, int(fnum))

    ws = pa.Workspace()
    ws.freq_grid = freq

    # 1)  Absorbing species + line catalogue, restricted to the freq window.
    ws.abs_speciesSet(species=species_from_tags(list(species)))
    ws.ReadCatalogData()
    ws.abs_bandsSelectFrequencyByLine(fmin=fmin, fmax=fmax)
    ws.WignerInit()
    ws.spectral_propmat_agendaAuto()

    # 2)  Atmosphere and surface.
    ws.surf_fieldPlanet(option="Earth")
    ws.atm_fieldRead(toa=100e3, basename=BNAME, missing_is_zero=1)
    t_surface = float(np.asarray(ws.atm_field["t"].data).flatten()[0])
    ws.surf_field[pa.arts.SurfaceKey("t")] = t_surface

    # 3)  Boundary conditions and the BT transform operator.
    ws.spectral_rad_space_agendaSet(option="UniformCosmicBackground")
    ws.spectral_rad_surface_agendaSet(option="Blackbody")
    ws.spectral_rad_transform_operatorSet(option="Tb")

    # 4)  Geometric ray path from the satellite down to the surface.
    ws.ray_pathGeometric(pos=[800e3, 0.0, 0.0], los=[180.0, 0.0],
                         max_stepsize=1000.0)

    # 5)  Jacobian target.
    ws.measurement_sensor = []
    ws.jac_targetsInit()
    if jacobian_quantity == "T":
        ws.jac_targetsAddTemperature()
    elif jacobian_quantity == "H2O":
        ws.jac_targetsAddSpeciesVMR(species="H2O")
        # NOTE:  ``jac_targetsToggleLogRelAtmTarget`` is a no-op in this build
        #  (ABS and LOG-REL give identical  ``spectral_rad_jac``).  The absolute
        #  -> relative VMR conversion is applied explicitly in step 7 below.
    else:
        raise ValueError(
            f"jacobian_quantity must be 'H2O' or 'T', got {jacobian_quantity!r}"
        )
    ws.jac_targetsFinalize()

    # 6)  Run the clear-sky emission.
    ws.spectral_radClearskyEmission()
    intensity = np.asarray(ws.spectral_rad)[:, 0]  # (n_freq,)
    bt = _bt_from_intensity(freq, intensity)

    # 7)  Per-level Jacobian in BT domain.
    J_int = np.asarray(ws.spectral_rad_jac)  # (n_alt, n_freq, 4 stokes)
    J_int_stokes0 = J_int[:, :, 0]
    dBT_dI = _dbtdI(freq, intensity)  # (n_freq,)
    J_bt = J_int_stokes0 * dBT_dI[None, :]  # (n_alt, n_freq)  [K per unit]

    #  ARTS3 always reports the species Jacobian in the ABSOLUTE-VMR domain
    #  (dI/dq).  ARTS 2.6 used  ``unit="rel"``, i.e. the RELATIVE (log-VMR)
    #  Jacobian  dI/dln(q) = q * (dI/dq).  In this build
    #  ``jac_targetsToggleLogRelAtmTarget`` is a no-op, so we perform the
    #  absolute -> relative conversion explicitly:  multiply each level by its
    #  H2O mixing ratio.  (No-op for the temperature target -- T is already an
    #  absolute quantity.)
    if jacobian_quantity == "H2O":
        q_h2o = np.asarray(ws.atm_field["H2O"].data).flatten()  # (n_alt,) VMR
        J_bt = J_bt * q_h2o[:, None]  # (n_alt, n_freq)  [K per unit-ln-VMR]

    # Normalise by altitude layer thickness in km   (2.6 convention).
    # 2.6  stores  ``jac``  as  ``(n_freq, n_alt)``  --  see  ``jac[freq_ind,  :]``.
    dk = np.gradient(alt / 1000.0)
    jac = (J_bt / dk[:, None]).T          # (n_freq, n_alt)

    # 8)  Cumulative opacity profile  $\\tau(z, z_{TOA})$  along the ray.
    ws.ray_pointBackground()
    ws.atm_pathFromPath()
    ws.freq_grid_pathFromPath()
    ws.spectral_propmat_pathFromPath()

    gp = ws.ray_path
    n_pts = len(gp)
    nf = len(freq)
    seg = np.asarray(ws.ray_path.distances(ws.surf_field.ellipsoid), dtype=float)
    coeff = np.zeros((n_pts, nf))
    for ip in range(n_pts):
        for jj in range(nf):
            coeff[ip, jj] = ws.spectral_propmat_path[ip][jj][0]

    #  Altitude of each ray point  [m];  the ray runs  top  ->  bottom,  so
    #  ``path_alt`` is monotonically  decreasing.
    path_alt = np.array([gp[ip].pos[0] for ip in range(n_pts)], dtype=float)

    #  Running  (trapezoidal) integral of the extinction coefficient, from
    #  the  top  of  the  ray  (``path_alt``[0]) down to each ray  point.
    cum = np.zeros(nf)
    tau_path = np.zeros((n_pts, nf))
    for ip in range(1, n_pts):
        cum += 0.5 * (coeff[ip - 1] + coeff[ip]) * seg[ip - 1]
        tau_path[ip] = cum

    #  ``tau_path``  is 0 at  the  top  and  (full zenith opacity) at the
    #  bottom.  Reverse  both  the  altitude  and  opacity arrays  so that
    #  ``np.interp``  sees  an  increasing  x-axis  (altitude  vs  the  tau
    #  accumulated  from the top down  to that  altitude).
    x = path_alt[::-1]                  # increasing altitude:  0  ->  top
    y = tau_path[:, :][::-1, :]         # decreasing opacity:   zenith  ->  0
    tau = np.array(
        [np.interp(a, x, y[:, jj]) for jj in range(nf) for a in alt]
    ).reshape(nf, -1).T                 # (n_alt, n_freq),  row 0 = surface
    tau = tau[::-1].copy()              # row 0 = TOA (tau = 0),  row -1 = zenith

    return {
        "freq": freq,
        "bt": bt,
        "tau": tau,
        "jac": jac,
        "alt": alt,
    }


# %% Run  module  as  script
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import os

    os.makedirs("plots", exist_ok=True)

    jacobian_quantity = "H2O"
    res = calc_jacobians(jacobian_quantity=jacobian_quantity)
    freq, bt, tau, jac, alt = (
        res["freq"], res["bt"], res["tau"], res["jac"], res["alt"]
    )
    print(f"frequency:   {freq.min()/1e9:.1f} -> {freq.max()/1e9:.1f}  GHz  "
          f"(n  =  {freq.size})")
    print(f"brightness  T:  {np.nanmin(bt):.2f}  K  ..  {np.nanmax(bt):.2f}  K")
    print(f"zenith   opacity:   {tau[-1].min():.2e}  ..  {tau[-1].max():.2e}")
    fi = argclosest(freq, 180e9)
    print(f"jacobian  peak (H2O, 180  GHz):  "
          f"{np.abs(jac[fi, :]).max():.4g}")
    assert np.all(np.isfinite(jac))
    assert np.max(np.abs(jac)) > 0

    highlight_frequency = 180e9  # Hz
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plot_brightness_temperature(freq, bt, where=highlight_frequency, ax=ax0)
    plot_opacity(freq, tau, where=highlight_frequency, ax=ax1)
    plot_jacobian(alt, jac[fi, :], jacobian_quantity=jacobian_quantity, ax=ax2)
    plot_opacity_profile(alt, tau[:, fi], ax=ax3)
    fig.tight_layout()
    fig.savefig(f"plots/jacobians-{jacobian_quantity}.pdf")
    plt.close(fig)
    print("All calculations  completed  successfully.")
