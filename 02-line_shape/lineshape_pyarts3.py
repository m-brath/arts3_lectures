# %% Import modules and define functions
"""Calculate and plot absorption cross sections (pyarts3)."""
import os
import re

import numpy as np
import pyarts3 as pa
import scipy as sp


def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


def linewidth(f, a):
    """Calculate the full-width at half maximum (FWHM) of an absorption line.

    Parameters:
        f (ndarray): Frequency grid.
        a (ndarray): Line properties
            (e.g. absorption coefficients or cross-sections).

    Returns:
        float: Linewidth.

    Examples:
        >>> f = np.linspace(0, np.pi, 100)
        >>> a = np.sin(f)**2
        >>> linewidth(f, a)
        1.571048056449009
    """

    idx = np.argmax(a)

    if idx < 3 or idx > len(a) - 3:
        raise RuntimeError(
            "Maximum is located too near at the edge.\n"
            + "Could not found any peak. \n"
            + "Please adjust the frequency range."
        )

    s = sp.interpolate.UnivariateSpline(f, a - np.max(a) / 2, s=0)

    zeros = s.roots()
    sidx = np.argsort((zeros - f[idx]) ** 2)

    if zeros.size == 2:
        logic = zeros[sidx] > f[idx]

        if np.sum(logic) == 1:
            fwhm = abs(np.diff(zeros[sidx])[0])

        else:
            print(
                "I only found one half maxima.\n"
                + "You should adjust the frequency range to have more reliable results.\n"
            )

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2

    elif zeros.size == 1:
        fwhm = abs(zeros[0] - f[idx]) * 2

        print(
            "I only found one half maxima.\n"
            + "You should adjust the frequency range to have more reliable results.\n"
        )

    elif zeros.size > 2:
        sidx = sidx[0:2]

        logic = zeros[sidx] > f[idx]

        print(
            "It seems, that there are more than one peak"
            + " within the frequency range.\n"
            + "I stick to the maximum peak.\n"
            + "But I would suggest to adjust the frequevncy range. \n"
        )

        if np.sum(logic) == 1:
            fwhm = abs(np.diff(zeros[sidx])[0])

        else:
            print(
                "I only found one half maxima.\n"
                + "You should adjust the frequency range to have more reliable results.\n"
            )

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2

    elif zeros.size == 0:
        raise RuntimeError(
            "Could not found any peak. :( \n"
            + "Probably, frequency range is too small.\n"
        )

    return fwhm


def calculate_absxsec(
    species="H2O",
    pressure=101325.0,
    temperature=300.0,
    fmin=175e9,
    fmax=190e9,
    fnum=1000,
    vmr=0.004,
):
    """Calculate absorption cross sections.

    Computes the *pure line-by-line* (Voigt) contribution for a single
    absorbing species.  The predefined/CIA (self-/cross-) continua are
    deliberately excluded so that the result is a clean, isolated absorption
    line, which is what the line-shape / FWHM analysis of this exercise
    requires.  (See ``AGENTS_MIGRATION.md`` for when to use the recipe vs.
    the manual line-by-line setup.)

    Parameters:
        species (str): Absorption species name (e.g. ``"H2O"``, ``"O3"``).
        pressure (float): Atmospheric pressure [Pa].
        temperature (float): Atmospheric temperature [K].
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.
        vmr (float): Volume mixing ratio of the absorbing species.

    Returns:
        ndarray, ndarray: Frequency grid [Hz], Abs. cross sections [m^2]
    """
    # 1. Ensure catalogue data and paths are available.
    pa.data.download()

    # 2. Set up a workspace restricted to the absorbing species and load the
    #    line-by-line (LBL) catalogue.
    ws = pa.Workspace()
    ws.abs_speciesSet(species=[species])
    ws.ReadCatalogData()
    ws.WignerInit()

    # 3. Keep only lines that lie inside the frequency window (speed-up) and
    #    define the frequency grid.
    ws.abs_bandsSelectFrequencyByLine(fmin=fmin, fmax=fmax)
    ws.freq_grid = np.linspace(fmin, fmax, fnum)

    # 4. Build a single atmospheric point and set its state.
    ws.atm_point["t"] = temperature
    ws.atm_point["p"] = pressure
    ws.atm_point[species] = vmr
    ws.ray_point.los = [180.0, 0.0]

    # 5. Compute ONLY the line-by-line Voigt contribution.
    #    We deliberately do NOT add the predefined/CIA (self-/cross-) continua
    #    here, so that the result is a clean, isolated absorption line -- which
    #    is what this exercise (line shape / FWHM) is about.  At high pressure
    #    the 183 GHz H2O line would otherwise sit on the microwave continuum
    #    tail and no isolated peak could be resolved.
    ws.jac_targetsInit()
    ws.spectral_propmatInit()
    ws.spectral_propmatAddVoigtLTE()

    # 6. Column 0 of the propagation matrix is the unpolarized absorption
    #    coefficient [1/m].  Convert it to a cross section using the number
    #    density of the absorbing species:  alpha = n * sigma.
    propmat = np.asarray(ws.spectral_propmat)
    number_density = ws.atm_point.number_density(species)
    absxsec = propmat[:, 0] / number_density
    freq = np.asarray(ws.freq_grid)

    return freq, absxsec


# %%  Run module as script
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    # Call ARTS to calculate absorption cross sections (several pressures)
    for pressure in (101325.0, 10132.5, 1013.25):
        freq, abs_xsec = calculate_absxsec(
            "H2O", pressure, 300, fmin=181e9, fmax=186e9, fnum=1000
        )

        assert abs_xsec.size > 0 and np.all(np.isfinite(abs_xsec))
        assert np.max(abs_xsec) > 0

        fig, ax = plt.subplots()
        ax.plot(freq / 1e9, abs_xsec)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel(r"Abs. cross section [$\sf m^2$]")
        ax.set_title(f"{tag2tex('H2O')} p:{pressure/100:.0f} hPa T:300 K")
        print(f"H2O @ {pressure:.0f} Pa: max xsec = {np.max(abs_xsec):.3e} m^2")
        fn = f"plots/plot_xsec_H2O_{pressure:.0f}Pa_300K.png"
        plt.savefig(fn)
        plt.close(fig)

    print("All calculations completed successfully.")
