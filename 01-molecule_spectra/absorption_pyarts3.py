# %% Import modules and define functions
"""Calculate and plot absorption cross sections (pyarts3)."""
import os
import re

import numpy as np
import pyarts3 as pa


def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


def calculate_absxsec(
    species="N2O",
    pressure=800e2,
    temperature=300.0,
    fmin=10e9,
    fmax=2000e9,
    fnum=10_000,
    vmr=0.01,
):
    """Calculate absorption cross sections.

    Uses the pyarts3 recipe ``pa.recipe.SingleSpeciesAbsorption``, which
    computes the *total* absorption: line-by-line (Voigt) plus the
    predefined/CIA (self-/cross-) continua.  This is the realistic case for
    single-band absorption studies, as required by this exercise.  (For
    isolated line-shape studies see ``02-line_shape``, which deliberately
    excludes the continua.)

    The recipe returns the unpolarized absorption coefficient ``alpha``
    [1/m] (first column of the propagation matrix).  It is converted to an
    absorption cross section ``sigma`` [m^2] using the number density ``n``
    of the absorbing species:  ``alpha = n * sigma``.

    Parameters:
        species (str): Absorption species name (e.g. ``"HCl"``, ``"CO2"``).
        pressure (float): Atmospheric pressure [Pa].
        temperature (float): Atmospheric temperature [K].
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.
        vmr (float): Volume mixing ratio of the absorbing species.  This is
                     mainly important for the water vapor continua.

    Returns:
        ndarray, ndarray: Frequency grid [Hz], Abs. cross sections [m^2]
    """
    # 1. Ensure catalogue data and paths are available (skipped if already
    #    downloaded or if ARTS_DATA_PATH is set).
    pa.data.download()

    # 2. The recipe sets up its own workspace (species selection, line
    #    catalogue, Wigner symbols, propmat agenda) for a single species.
    recipe = pa.recipe.SingleSpeciesAbsorption(species=species)

    # 3. Define the frequency grid.
    freq = np.linspace(fmin, fmax, fnum)
    f_grid = pa.arts.AscendingGrid(freq)

    # 4. Build a single atmospheric point and set its state, including the
    #    volume mixing ratio of the absorbing species.
    p = pa.arts.AtmPoint()
    p.temperature = temperature
    p.pressure = pressure
    p.wind = np.array([0.0, 0.0, 0.0])
    p[species] = vmr

    # 5. Run the recipe.  The return value is the first column of the
    #    propagation matrix, i.e. the unpolarized absorption coefficient
    #    alpha [1/m] (the full matrix is ws.spectral_propmat, shape (nf, 7)).
    alpha = np.asarray(recipe(f_grid, p))

    # 6. Convert absorption coefficient to cross section: sigma = alpha / n.
    number_density = p.number_density(species)
    absxsec = alpha / number_density

    return freq, absxsec


# %%  Run module as script
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    # Define parameters
    species = "N2O"
    pressure = 1000e2
    temperature = 300.0
    fmin = 1e12
    fmax = 1e14
    fnum = 10_000

    # Calculate absorption cross sections
    freq, absxsec = calculate_absxsec(
        species, pressure, temperature, fmin, fmax, fnum
    )

    assert absxsec.size > 0 and np.all(np.isfinite(absxsec))

    # Plot absorption cross sections
    fig, ax = plt.subplots()
    ax.semilogy(freq, absxsec)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Abs. cross section [$\sf m^2$]")
    ax.set_title(f"{tag2tex(species)} p:{pressure/100:.0f} hPa T:{temperature:.0f} K")
    print(f"{species}: max xsec = {np.max(absxsec):.3e} m^2")
    plt.savefig("plots/plot_xsec_N2O.png")
    print("Plot saved to plots/plot_xsec_N2O.png")
