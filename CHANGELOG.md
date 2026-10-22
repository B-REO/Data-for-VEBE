# Changelog

## [2.0.0] - 2026-10-20

Release corresponding to the revised manuscript.

### Added
- Simulation results and notebooks for the approximate shift (Figs. 8–13, Appendix C).
- `scripts/module/analysis.py` providing common analysis functions for VEBE and VQE.
- Loading functions for the optimization results in `scripts/module/loading_tools.py`.
- `LICENSE` (MIT, code) and `LICENSE-DATA` (CC BY 4.0, data).
- Correspondence table between figures, notebooks, and scripts in `README.md`.

### Changed
- Reorganized the data directories by method and shift type (`exact` / `approximate`).
- Rewrote the analysis notebooks for the hydrogen chains and N2 so that the exact and approximate shifts are analyzed with the same functions.
- Updated equation numbers in comments to match the revised manuscript.
- Updated `CITATION.cff`, `README.md`, and `requirements.txt`.

### Fixed
- The public VEBE simulation scripts now block-encode the shifted Hamiltonian (coefficients after subtracting the shift and delta), consistent with the code used to generate the data in the manuscript.
- `optimal_shots_for_vqe_terms` now raises an error when the numbers of coefficients and Pauli terms do not match, instead of silently truncating them.
- Removed a debug print statement in `sampling.py`.

## [1.1.0] - 2026-06-20

### Added
- Added scripts to generate the data used in Fig. 7.
- Added new functions to modules used by the above scripts.
- Added reproducible notebooks for Fig. 7.
- Added a new directory for storing PNG files.
- Added an option to select the output file format.
- Added data required to reproduce Fig. 7.

### Changed
- Refactored optimization scripts into modules.
- Improved documentation and type annotations.

### Fixed
- Fixed path resolution issues in Jupyter notebooks.

## [1.0.0] - 2026-01-26

### Added
- Initial public release.
- Simulation scripts for VQE and VEBE.
- Data and notebooks corresponding to Figs. 3–6.