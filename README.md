# MOLecular Lines Identification

Automated Gaussian fitting of spectra and identification of molecular emission lines with the JPL and CDMS catalogues.


## Files

The pipeline has two independent stages (aka two files):

| Script | Purpose |
|---|---|
| `gauss_fit_new.py` | Iterative Gaussian decomposition of a 1D spectrum |
| `match_isotopes_isomers_o.py` | Matching lines with the JPL/CDMS databases and outputs a LaTeX results table |

---

## Requirements

### Python version
Python 3.8 or newer.

### Python packages
Install all dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
numpy
matplotlib
scipy
lmfit
uncertainties
scienceplots
tqdm
pandas
scikit-learn
```

> **Note:** `scienceplots` requires LaTeX for full functionality. If LaTeX is not installed on your system, replace `['science', 'no-latex']` with `['science', 'no-latex']` (already set in the code) — no further change is needed, but make sure `scienceplots` itself is installed.

### External databases (for `match_isotopes_isomers_o.py` only)

The matcher queries two local SQLite databases built from the JPL and CDMS spectral catalogues:

| Variable in code | Description |
|---|---|
| `myjpl.db` | JPL catalogue as a SQLite database |
| `my_cdms.db` | CDMS catalogue as a SQLite database |


Two plain-text molecule lists are also required:

| File | Description |
|---|---|
| `ism_mol.dat` | One molecule name per line — the ISM molecule reference list |
| `iso_izo.dat` | One molecule name per line — the isotopologue reference list |
| `species_united.dat` | One molecule name per line — secondary/extended species list |

---

## Usage

### Stage 1 — Gaussian fitting (`gauss_fit_new.py`)

Edit the `__main__` block at the bottom of the file:

```python
# Path to the input spectrum (two-column whitespace-separated: frequency[MHz]  intensity[K])
name = 'path/to/your_spectrum.dat'

# Spectral line width in km/s (initial guess)
dv = 1.0

models = fit_spectrum_with_gaussians(
    obsX, obsY,
    threshold=0.001,        # Minimum peak flux to attempt a fit (same units as spectrum)
    line_width_kms=dv,
    max_iterations=1000,    # Maximum number of Gaussians to fit
    rest_freq=205.000988    # Rest frequency of the target transition in GHz
)
```

Then run:

```bash
python gauss_fit_new.py
```

**Outputs:**

- `intermediate_plots_lmfit/final_fit_lmfit_diagnostic.png` — spectrum with all fitted Gaussians overlaid, residuals, and amplitude vs. line-width scatter plot
- `intermediate_plots_lmfit/final_fit_lmfit_statistics.png` — histograms of line widths, amplitudes, SNR, and fit quality
- A `.dat` file (path set in `save_results()`) with amplitude, centre frequency, FWHM width, and integrated area for every fitted Gaussian, all with uncertainties

### Stage 2 — Molecular line matching (`match_isotopes_isomers_o.py`)

Edit the `main()` function near the top:

```python
rest_freq = 243000          # Rest frequency in MHz

# Path to the Gaussian fit output produced by gauss_fit_new.py
params_file = 'path/to/your_gauss_output.dat'

# Desired output paths
output_file    = 'path/to/matched_lines.dat'
candidates_file = 'path/to/candidates.dat'
latex_output_file = 'path/to/results_table.tex'
```



Then run:

```bash
python match_isotopes_isomers_o.py
```

**Outputs:**

- `matched_lines.dat` — best molecular match for each observed line
- `candidates.dat` — all candidate matches ranked by score
- `results_table.tex` — ready-to-compile LaTeX table of identified lines with frequencies, upper-level energies, integrated intensities, line widths, peak temperatures, v_LSR, and catalogue source

---

## Key parameters

### `fit_spectrum_with_gaussians`

| Parameter | Description |
|---|---|
| `threshold` | Flux cutoff below which no new Gaussian is attempted |
| `line_width_kms` | Expected line width in km/s — used as the initial sigma guess |
| `max_iterations` | Hard cap on the total number of fitted components |
| `rest_freq` | Rest frequency in GHz, used to convert frequency widths to km/s |

### `EnhancedMolecularLineMatcher`

| Parameter | Description |
|---|---|
| `vlsr` | Source LSR velocity in km/s — applied as a frequency shift before matching |
| `confidence_level` | Minimum score threshold (0–1) for a match to be accepted |
| `frequency_window` | Initial search window around each line centre in GHz |

---

## Input spectrum format

`gauss_fit_new.py` expects a two-column ASCII file (whitespace-separated, no header):

```
<frequency_MHz>   <intensity_K>
<frequency_MHz>   <intensity_K>
...
```

The script multiplies frequency values by `1e-3` to convert MHz → GHz internally, so supply the file in MHz.

---

## Notes

- The matcher first tries CDMS, then falls back to JPL for any lines that remain unmatched.
- Matching priority order: primary species list → secondary species list (`species_united.dat`) → ISM molecules list (`ism_mol.dat`).
- The LaTeX table filters to lines with amplitude ≥ 0.016 K — adjust `filtered_df = best_matches_df[best_matches_df['amplitude'] >= 0.016]` in `main()` if needed.
- Hardcoded output paths in `save_results()` inside `gauss_fit_new.py` should be updated before first use.
