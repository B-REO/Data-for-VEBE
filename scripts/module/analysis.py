"""
Analysis utilities for VEBE and VQE simulation results.

This module provides functions for evaluating statistical quantities from
precomputed VEBE and VQE simulation data and for estimating the theoretical
minimum number of measurements required to achieve a specified target
standard deviation.

Functions
---------
analyze_vebe
    Analyze sampled VEBE energies and estimate the theoretical measurement
    cost for each molecular geometry.

analyze_vqe
    Analyze sampled VQE energies and estimate the theoretical measurement
    cost for each molecular geometry.

extreme_eigenvalues
    Compute the minimum and maximum eigenvalues of a qubit Hamiltonian.

Notes
-----
The target standard deviation supplied to the analysis functions is expressed
in Hartree. Hamiltonian construction is handled by utilities in
``module.loading_tools``.
"""

import numpy as np
from openfermion.linalg import get_sparse_operator 
from scipy.sparse.linalg import eigsh
from module.loading_tools import qubit_hamiltonian

def analyze_vebe(rhf_files, npz_files, target_sd):
    """
    Analyze VEBE simulation results for multiple molecular geometries.

    For each geometry, this function loads sampled VEBE energy estimates
    and the Hamiltonian shift used in the simulation. It calculates the
    mean energy, unbiased sample standard deviation, extremal eigenvalues,
    an upper bound on the maximum eigenvalue, and the theoretical minimum
    number of measurements required to achieve the specified target
    standard deviation.

    Parameters
    ----------
    rhf_files : list of tuple
        RHF-file information. The first element of each tuple must be the
        path to the corresponding RHF file.
    npz_files : list of str or pathlib.Path
        Paths to NPZ files containing VEBE simulation results. ``array1``
        must contain sampled shifted-energy estimates and ``array2`` must
        contain the Hamiltonian shift used in the simulation.
    target_sd : float
        Target standard deviation in Hartree.

    Returns
    -------
    dict
        Dictionary containing the following NumPy arrays:

        - ``energy``: Mean VEBE energies after restoring the shift.
        - ``sd``: Unbiased sample standard deviations.
        - ``t_min``: Theoretical minimum numbers of measurements.
        - ``gamma``: Hamiltonian shifts used in the simulations.
        - ``lam_min``: Minimum eigenvalues of the Hamiltonians.
        - ``lam_max``: Maximum eigenvalues of the Hamiltonians.
        - ``lam_max_ub``: Upper bounds on the maximum eigenvalues.

    Notes
    -----
    The block-encoding normalization factor and success probability are
    evaluated using the shift stored in each simulation-result file.
    """
    out = {k: [] for k in ("energy", "sd", "t_min", "gamma", "lam_min", "lam_max", "lam_max_ub")}
    for rhf, npz in zip(rhf_files, npz_files):
        coeffs, _, jw_h = qubit_hamiltonian(rhf[0])
        res = np.load(str(npz))
        samples, gamma = res["array1"], float(res["array2"])  # gamma = shift used in the run

        out["energy"].append(np.mean(samples) + gamma)
        out["sd"].append(np.sqrt(np.var(samples, ddof=1)))
        out["gamma"].append(gamma)

        e_min, e_max = extreme_eigenvalues(jw_h)
        out["lam_min"].append(e_min)
        out["lam_max"].append(e_max)
        out["lam_max_ub"].append(coeffs[0] + np.sum(np.abs(coeffs[1:])))  # Eq. (26)

        shifted = coeffs.copy()
        shifted[0] -= gamma
        s_prime = np.sum(np.abs(shifted))
        # p_succ of the ground state for THIS gamma (not the exact midpoint)
        p_succ = (e_min - gamma) ** 2 / s_prime**2
        out["t_min"].append(s_prime**2 * (1 - p_succ) / (4 * (target_sd) ** 2))
    return {k: np.array(v) for k, v in out.items()}

def analyze_vqe(rhf_files, npz_files, target_sd):
    """
    Analyze VQE simulation results for multiple molecular geometries.

    For each geometry, this function reconstructs sampled VQE energies
    from the sampled expectation values of individual Pauli terms. It then
    calculates the mean energy, unbiased sample standard deviation, and
    theoretical minimum number of measurements required to achieve the
    specified target standard deviation.

    Parameters
    ----------
    rhf_files : list of tuple
        RHF-file information. The first element of each tuple must be the
        path to the corresponding RHF file.
    npz_files : list of str or pathlib.Path
        Paths to NPZ files containing VQE sampling results in ``array1``.
        Each Pauli-term result is expected to contain the sampled
        expectation value and theoretical probability used in the
        measurement-cost calculation.
    target_sd : float
        Target standard deviation in Hartree.

    Returns
    -------
    dict
        Dictionary containing the following NumPy arrays:

        - ``energy``: Mean sampled VQE energies.
        - ``sd``: Unbiased sample standard deviations.
        - ``t_min``: Theoretical minimum numbers of measurements.
    """
    out = {k: [] for k in ("energy", "sd", "t_min")}
    for rhf, npz in zip(rhf_files, npz_files):
        coeffs, _, _ = qubit_hamiltonian(rhf[0])
        samples = np.load(str(npz))["array1"]
        estimates = [np.sum(coeffs * s[:, 1]) for s in samples]
        out["energy"].append(np.mean(estimates))
        out["sd"].append(np.sqrt(np.var(estimates, ddof=1)))
        p = samples[0][:, 2]
        a = np.sum(2 * np.abs(coeffs) * np.sqrt(np.abs(p * (1 - p))))  # Eq. 9
        out["t_min"].append(a**2 / (target_sd) ** 2)
    return {k: np.array(v) for k, v in out.items()}

def extreme_eigenvalues(jw_h):
    """
    Compute the minimum and maximum eigenvalues of a qubit Hamiltonian.

    The Hamiltonian is converted to a sparse matrix and its algebraically
    smallest and largest eigenvalues are calculated using
    ``scipy.sparse.linalg.eigsh``.

    Parameters
    ----------
    jw_h
        Jordan-Wigner-transformed qubit Hamiltonian.

    Returns
    -------
    e_min : float
        Minimum eigenvalue.
    e_max : float
        Maximum eigenvalue.
    """
    h = get_sparse_operator(jw_h).tocsr()
    e_min = float(eigsh(h, k=1, which="SA")[0][0])
    e_max = float(eigsh(h, k=1, which="LA")[0][0])
    return e_min, e_max
