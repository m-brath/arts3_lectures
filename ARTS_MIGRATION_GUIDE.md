# ARTS Migration Guide: pyarts $\rightarrow$ pyarts3

This document provides a guide and reference for migrating ARTS exercises from the `pyarts` (2.6) framework to the `pyarts3` framework.

## 1. Core Concept Changes

The shift from `pyarts` to `pyarts3` involves moving from a purely manual workspace configuration towards a **Recipe-based** model. 

- **`pyarts` (2.6):** You manually configure every component of the `Workspace` (species, pressure, temperature, line shapes) and then call the relevant method.
- **`pyarts3`:** While you can still manipulate the workspace directly, the standard and recommended approach is to use **Recipes** (e.g., `pa.recipe.SingleSpeciesAbsorption`). Recipes encapsulate a sequence of workspace methods (an "agenda") to simplify complex setups.

## 2. Conversion Patterns

| Task | `pyarts` (2.6) approach | `pyarts3` approach |
| :--- | :--- | :--- |
| **Data Initialization** | Manual/Implicit | `pa.data.download()` (Ensures paths are set) |
| **Species Setting** | `ws.abs_speciesSet(species=['...'])` | Handled by recipe or `ws.abs_speciesSet` |
| **Atmosphere Setup** | `ws.rtp_pressure = ...`, `ws.rtp_temperature = ...` | Create an `pa.arts.AtmPoint()` and set its attributes |
| **VMR (Mixing Ratio)** | `ws.rtp_vmr = np.array([vmr])` | `p[species] = vmr` (on an `AtmPoint` object) |
| **Frequency Grid** | `ws.VectorNLinSpace(ws.f_grid, ...)` | `pa.arts.AscendingGrid(np.linspace(...))` |
| **Execution** | `ws.MethodName()` | `recipe(f_grid, atm_point)` |
| **Result Retrieval** | Accessing `ws.variable.value` | Accessing `recipe.ws.variable.value` (returns numpy array) |

## 3. Migration Template (Python)

Use this template as a starting point for new ported scripts.

```python
import numpy as np
import pyarts3 as pa

def calculate_task(
    species="N2O",
    pressure=800e2,
    temperature=300.0,
    fmin=10e9,
    fmax=2000e9,
    fnum=100
):
    try:
        # 1. Essential: Set data paths and download
        pa.data.download()
        
        # 2. Initialize the recipe
        recipe = pa.recipe.SingleSpeciesAbsorption(species=species)
        
        # 3. Set up the Atmosphere Point
        p = pa.arts.AtmPoint()
        p.temperature = temperature
        p.pressure = pressure
        p.wind = np.array([0.0, 0.0, 0.0])
        p[species] = 0.01 # Set VMR
        
        # 4. Define Frequency Grid
        f_grid = pa.arts.AscendingGrid(np.linspace(fmin, fmax, fnum))

        # 5. Execute calculation via recipe
        recipe(f_grid, p)
        
        # 6. Extract results from the recipe's workspace
        # The first dimension is freq, second is variables (0 = unpolarized absorption)
        res_matrix = recipe.ws.spectral_propmat.value
        results = res_matrix[:, 0]
        
        return f_grid.value, results
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    f_grid, absxsec = calculate_task()
    if absxsec is not None:
        plt.semilogy(f_grid, absxsec)
        plt.show()
```
