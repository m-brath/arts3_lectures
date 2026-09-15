import numpy as np
import pyarts3 as pa

def calculate_absxsec(
    species="N2O",
    pressure=800e2,
    temperature=300.0,
    fmin=10e9,
    fmax=2000e9,
    fnum=100
):
    try:

        pa.data.download()

        # Using the recipe approach for simplicity and following pyarts3 best practices
        recipe = pa.recipe.SingleSpeciesAbsorption(species=species)
        
        # Create the atmosphere point
        p = pa.arts.AtmPoint()
        p.temperature = temperature
        p.pressure = pressure
        p.wind = np.array([0.0, 0.0, 0.0])
        # Set the VMR for the requested species
        p[species] = 0.01
        
        # Create frequency grid
        f_grid = pa.arts.AscendingGrid(np.linspace(fmin, fmax, fnum))

        # The recipe handles its own workspace
        recipe(f_grid, p)
        
        # The recipe returns the spectral_propmat
        res_matrix = recipe.ws.spectral_propmat.value
        
        # The recipe returns a numpy array.
        # We assume the first dimension is the size of the frequency grid 
        # and the second dimension contains 7 variables, the first of which is unpolarized absorption.
        absxsec_raw = res_matrix[:, 0]
        
        return f_grid.value, absxsec_raw
    
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define parameters
    species = "N2O"
    pressure = 1000e2
    temperature = 300.0
    fmin = 1e12
    fmax = 1e14
    fnum = 10_000

    # Calculate absorption cross sections
    f_grid, absxsec = calculate_absxsec(
        species, pressure, temperature, fmin, fmax, fnum
    )

    # Plot absorption cross sections
    if absxsec is not None:
        fig, ax = plt.subplots()
        ax.semilogy(f_grid, absxsec)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"Abs. (raw value) [$\sf m^2$]")
        plt.savefig("plot.png")
        print("Plot saved to plot.png")
    else:
        print("No absorption cross section data returned.")
