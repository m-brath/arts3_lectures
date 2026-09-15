# AGENTS_MIGRATION.md — Protocol for ARTS Migration (pyarts $\rightarrow$ pyarts3)

This document is an instruction set for AI agents tasked with porting Python exercises from the `pyarts` (2.6) framework to the `pyarts3` framework.

## 1. Mission Objective

The goal is to migrate Python exercises from the legacy `pyarts` framework to the modern `pyarts3` framework, ensuring functional equivalence and adhering to the new "Recipe-based" paradigm.

**IMPORTANT:** Every exercise includes a Jupyter Notebook (`.ipynb`). The notebook is the primary interface for students during lectures. **The migration of the notebook is a critical part of the task and must be completed alongside the Python scripts.**

## 2. Standard Operating Procedure (SOP)

### Step 1: Analysis
- Identify all `import pyarts` or `import pyarts.xxx` statements.
- Locate the core calculation logic (usually a function like `calculate_absxsec`).
- Identify the input parameters (pressure, temperature, species, etc.).
- **Identify the associated Jupyter Notebook (`.ipynb`) and its requirements (e.g., what functions it imports from the Python script).**

### Step s: Script Refactoring (The "Recipe" Pattern)

When converting, **always** prefer the `pyarts3.recipe` approach over manual workspace manipulation.

**Mandatory Changes:**
1.  **Data Initialization:** Always insert `pa.data.download()` at the start of the calculation function.
2.  **Recipe Usage:** Replace manual `ws` setup with the appropriate `pa.recipe` method (e.g., `pa.recipe.SingleSpeciesAbsorption(species=species)`).
3.  **Atmosphere Setup:**
    - Do **not** use `ws.AtmosphereSet1d()` or similar manual workspace calls.
    - Instead, create an `pa.arts.AtmPoint()` object.
    - Set properties directly on the object: `p.temperature = ...`, `p.pressure = ...`, `p.wind = ...`.
    - **Crucial:** Set the species concentration using the dictionary-like syntax: `p[species] = vmr_value`.
4.  **Frequency Grid:** Use `pa.arts.AscendingGrid(np.linspace(...))` instead of workspace-based vector creation.

### Step 3: Notebook Migration
- Port the markdown cells from the legacy notebook to the new notebook.
- Update the code cells in the notebook to import from the newly ported `.py` file instead of the old one.
- Ensure the notebook's execution environment (imports and data availability) is compatible with the new `pyarts3` implementation.

### Step 4: Result Extraction

`pyarts3` results often wrap NumPy arrays. 
- If the result is a workspace variable, access the raw array using `.value`.
- For recipes, the return value is typically a `numpy.ndarray` directly.

## 3. Known Pitfalls (DO NOT IGNORE)

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| **AttributeError** | Attempting to call `ws.AtmosphereSet1D()` or `ws.abs_lines_...` | These are often deprecated or internal. Use `pa.recipe` or configure `AtmPoint`. |
| **RuntimeError (Species VMR not found)** | Forgetting to set the species concentration on the `AtmPoint` object. | Use `p[species] = value` after creating the `AtmPoint`. |
| **AttributeError (ndarray has no attribute 'value')** | Trying to call `.value` on an object that is already a NumPy array. | Check `type(result)` before calling `.value`. |
| **RuntimeError (XML parse error)** | Missing or malformed catalog files in `lines/`. | Ensure `pa.data.download()` is called and directories are correctly linked. |

## 4. Verification Protocol

Every migrated script must pass this test sequence:
1.  **Syntax Check:** `python -m py_compile <file>.py`
2.  **Functional Test:** Run the script in a standard environment.
3.  **Notebook Check:** Ensure the `.ipynb` file correctly imports the ported `.py` module and the calculation executes without error.
4.  **Success Criteria:** 
    - Script completes without `RuntimeError`.
    - The result is a non-empty NumPy array.
    - The plot (if generated) contains non-zero/non-NaN values.
