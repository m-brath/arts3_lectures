# Exercise 5: Inversion theory (pyarts3)

This port demonstrates a linear Optimal Estimation Method retrieval of the
H2O profile from a simulated, upward-looking 22 GHz water-vapor radiometer.
The profile and a priori covariance retain the legacy exercise's 48 retrieval
levels.

Use the repository's `srs` environment and run commands from this directory:

```sh
mamba run -n srs python simulate_measurement_pyarts3.py
mamba run -n srs python oem_pyarts3.py
```

The generator writes deterministic synthetic data as ARTS 3 `GriddedField1`
XML files to `input/measurement_arts3.xml` and `input/x_true_arts3.xml`. The
notebooks only read those generated files; they do not overwrite them. Run the
student notebook after generating the data, and use the solution notebook to
see completed retrieval, gain-matrix, and averaging-kernel calculations.

The ARTS 3 port uses its available split catalog and continuum data rather than
the exact ARTS 2.6 CKDMT/TRE05 selections. At three sample channels around the
22.235 GHz line, the ARTS 3 brightness temperatures were about 0.197 K (5.2%)
lower than ARTS 2.6; the Jacobian relative norm difference was about 1.3%.
Regenerated measurements are internally consistent with the ARTS 3 forward
model, so do not compare retrieval residuals directly with the legacy
measurement.
