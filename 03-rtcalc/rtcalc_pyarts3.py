# %% Import modules and define functions
"""Calculate and plot zenith opacity and brightness temperatures (pyarts3)."""
import re

import numpy as np
import pyarts3 as pa

# Physical constants (SI)
H = 6.62607015e-34  # Planck constant [J s]
C = 299792458.0  # speed of light [m/s]
K_B = 1.380649e-23  # Boltzmann constant [J/K]


def tags2tex(tags):
    """Replace all numbers in every species tag with LaTeX subscripts."""
    return [re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag) for tag in tags]


def species_from_tags(tags):
    """Translate the lecture species tags to ARTS3 absorption species.

    ARTS3 resolves bare element tags (``H2O``, ``O2``, ``N2``) to the
    ``lines/`` family in  ``arts - cat - data - trunk``, matching the
    ``ReadSpeciesSplitCatalog(basename="lines/")``  bundle that the  2.6
    lecture uses.  The  ``-PWR2022`` /  ``-CKDMT``  suffixed tags are
    separate, much heavier, line sets that are NOT  the  2.6 reference data.

    Parameters:
        tags (list[str]): Lecture species tags (e.g. ``["N2", "O2", "H2O"]``).

    Returns:
        list[str]: ARTS3 absorption species tags.
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


def planck_brightness_temperature(freq, intensity):
    """Invert the Planck formula to get brightness temperature.

    Brightness temperature is the temperature of a blackbody that emits the
    same spectral intensity.  ARTS reports the upwelling intensity in units of
    radiance per Hz, so the inverse Planck relation below recovers the
    temperature for a given frequency.

    Parameters:
        freq (ndarray): Frequency grid [Hz].
        intensity (ndarray): Spectral intensity [W m^-2 sr^-1 Hz^-1].

    Returns:
        ndarray: Brightness temperature [K].
    """
    # Intensity B = (2 h f^3 / c^2) / (exp(h f / kB T) - 1)
    # => T = (h f / kB) / log(1 + 2 h f^3 / (c^2 B))
    hf_kb = H * freq / K_B
    pref = 2.0 * H * freq**3 / (C**2 * intensity)
    return hf_kb / np.log1p(pref)


def run_arts(
    species,
    zenith_angle=0.0,
    height=0.0,
    fmin=10e9,
    fmax=250e9,
    fnum=1_000,
):
    """Perform a clear-sky radiative transfer simulation.

    Brightness temperature [K] and zenith opacity (optical depth) [1] are
    computed for the given observing geometry.

    Parameters:
        species (list[str]): List of species tags (e.g. ``["N2", "O3", "H2O"]``).
        zenith_angle (float): View angle [deg]; 0 = up, 180 = down (nadir).
        height (float): Observer height [m] (nadir look from a satellite
            uses ``height=800e3`` and ``zenith_angle=180``).
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.

    Returns:
        ndarray, ndarray, ndarray:
          Frequency grid [Hz], Brightness temperature [K], Optical depth [1]
    """
    # 1. Ensure catalogue data and profile paths are available.
    pa.data.download()

    freq = np.linspace(fmin, fmax, fnum)

    ws = pa.Workspace()
    ws.freq_grid = freq

    # 2. Species + line-by-line catalogue, restricted to the frequency window.
    ws.abs_speciesSet(species=species_from_tags(species))
    ws.ReadCatalogData()
    ws.abs_bandsSelectFrequencyByLine(fmin=fmin, fmax=fmax)
    ws.WignerInit()

    # 3. Automatic propagation-matrix (absorption) agenda for the lines.
    ws.spectral_propmat_agendaAuto()

    # 4. Planet, atmosphere and surface.  The midlatitude-summer profile is
    #    read in the ARTS3 file layout; the surface temperature is taken from
    #    the surface layer of the profile itself (wet-land / smooth surface).
    ws.surf_fieldPlanet(option="Earth")
    ws.atm_fieldRead(
        toa=100e3, basename="planets/Earth/afgl/midlatitude-summer/", missing_is_zero=1
    )
    t_surface = float(np.asarray(ws.atm_field["t"].data).flatten()[0])
    ws.surf_field[pa.arts.SurfaceKey("t")] = t_surface

    # 5. Boundary conditions at the ends of the ray:
    #    - a ray ending in space is backed by the cold cosmic background,
    #    - a ray ending at the ground is backed by the surface (blackbody).
    ws.spectral_rad_space_agendaSet(option="UniformCosmicBackground")
    ws.spectral_rad_surface_agendaSet(option="Blackbody")
    ws.spectral_rad_transform_operatorSet(option="Tb")

    # 6. Observation geometry.  ARTS3 takes observer position as [alt, lat, lon]
    #    and the ray is built down/up along the line of sight by the RT solver.
    ws.ray_pathGeometric(
        pos=[float(height), 0.0, 0.0],
        los=[float(zenith_angle), 0.0],
        max_stepsize=1000.0,
    )

    # 7. Solve the clear-sky emission along the path.
    ws.spectral_radClearskyEmission()
    intensity = np.asarray(ws.spectral_rad)[:, 0]
    brightness_temperature = planck_brightness_temperature(freq, intensity)

    # 8. Zenith opacity: integrate the absorption coefficient along the same
    #    path (trapezoidal rule), summing layer by layer.
    ws.ray_pointBackground()
    ws.atm_pathFromPath()
    ws.freq_grid_pathFromPath()
    ws.spectral_propmat_pathFromPath()

    n_path = len(ws.ray_path)
    n_freq = len(freq)
    coeff = np.array(
        [[ws.spectral_propmat_path[ip][ff][0] for ff in range(n_freq)] for ip in range(n_path)]
    )
    segment = np.asarray(ws.ray_path.distances(ws.surf_field.ellipsoid), dtype=float)
    d_tau = 0.5 * (coeff[:-1] + coeff[1:]) * segment[:, None]
    optical_depth = d_tau.sum(axis=0)

    return freq, brightness_temperature, optical_depth


# %% Run module as script
if __name__ == "__main__":
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    species = ["N2", "O2", "H2O"]
    height = 0.0  # m
    zenith_angle = 0.0  # deg

    # Run the radiative transfer simulation to get the frequency grid,
    # brightness temperature and optical depth
    freq, bt, od = run_arts(species, zenith_angle, height)

    assert od.size > 0 and np.all(np.isfinite(od))
    assert np.max(od) > 0

    print(f"frequency: {freq.min()/1e9:.1f} -> {freq.max()/1e9:.1f} GHz, n={freq.size}")
    print(f"opacity:   {np.min(od):.2e} .. {np.max(od):.2e}")
    print(f"brightness T: {np.nanmin(bt):.2f} K .. {np.nanmax(bt):.2f} K")

    # Plot the zenith opacity with logarithmic scale on y axis
    fig, ax = plt.subplots()
    ax.semilogy(freq / 1e9, od)
    ax.axhline(1, linewidth=0.8, color="#b0b0b0", zorder=0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Zenith opacity")
    ax.set_title(f"{', '.join(tags2tex(species))}")
    plt.savefig("plots/opacity.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot the brightness temperature
    fig, ax = plt.subplots()
    ax.plot(freq / 1e9, bt)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Brightness temperature [K]")
    ax.set_title(f"{', '.join(tags2tex(species))}")
    plt.savefig("plots/brightness_temperature.pdf", bbox_inches="tight")
    plt.close(fig)

    print("All calculations completed successfully.")
