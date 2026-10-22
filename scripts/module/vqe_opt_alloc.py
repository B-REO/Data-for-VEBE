"""
Variance-based measurement-shot allocation for VQE.

This module provides utilities for calculating the exact measurement
variance of individual Pauli terms from a simulated quantum state and
allocating a fixed total number of measurement shots among those terms.

The allocation follows the variance-weighted rule

    t_i ∝ |c_i| sqrt(Var[P_i]),

where ``c_i`` is the coefficient of Pauli term ``P_i``.

Functions
---------
estimate_pauli_variance_from_state
    Calculate the measurement variance of a Pauli operator from a Qulacs
    quantum state.

optimal_shots_for_vqe_terms
    Allocate a fixed measurement budget among Pauli terms according to
    their coefficients and measurement variances.

Notes
-----
Expectation values are evaluated exactly using Qulacs. The resulting
variances are therefore theoretical simulator quantities rather than
sampling-based estimates.
"""

import numpy as np
import re
from qulacs import QuantumState, Observable

def _compact_to_qulacs_pauli_string(term: str) -> str:
    """
    Convert a compact Pauli string to Qulacs observable syntax.

    Parameters
    ----------
    term : str
        Compact Pauli string such as ``"X0X1Y2Y3"``. An empty string
        represents the identity operator.

    Returns
    -------
    str
        Qulacs-formatted Pauli string. For example,
        ``"X0Y2"`` is converted to ``"X 0 Y 2"``.
    """
    if not term:
        return ""
    parts = []
    for p, idx in re.findall(r'([XYZ])(\d+)', term):
        parts.append(f"{p} {idx}")
    return " ".join(parts)

def estimate_pauli_variance_from_state(term: str, n_qubits: int, state: QuantumState) -> float:
    """
    Calculate the measurement variance of a Pauli operator.

    For a Pauli operator ``P`` with measurement outcomes +1 and -1, the
    variance is evaluated as ``1 - <P>**2`` using the exact expectation
    value of the supplied quantum state.

    Parameters
    ----------
    term : str
        Compact Pauli string such as ``"X0Y1Z2"``. An empty string
        represents the identity operator.
    n_qubits : int
        Total number of qubits.
    state : QuantumState
        Qulacs quantum state used to evaluate the expectation value.

    Returns
    -------
    float
        Measurement variance of the Pauli operator. The identity operator
        has zero variance.

    Notes
    -----
    Variances smaller than ``1e-12`` are treated as zero to suppress
    numerical round-off errors.
    """
    if not term:
        return 0.0
    obs = Observable(n_qubits)
    obs.add_operator(1.0, _compact_to_qulacs_pauli_string(term))
    expv = float(obs.get_expectation_value(state))
    var = 1.0 - expv * expv
    if var < 1.e-12:
        return 0.0
    return float(var) if var > 0.0 else 0.0

def optimal_shots_for_vqe_terms(
    re_num,
    re_ope,
    state,
    n_qubits,
    total_shots,
    min_shots_nonzero: int = 1,
    round_mode: str = "int",     
    enforce_total: bool = True,  
):
    """
    Allocate measurement shots among VQE Pauli terms.

    The allocation follows the variance-weighted rule
    ``t_i proportional to |c_i| * sqrt(Var[P_i])``,
    where ``c_i`` is the coefficient of Pauli term ``P_i``. Exact
    expectation values of the supplied simulator state are used to
    calculate the Pauli variances.

    Parameters
    ----------
    re_num : array_like
        Coefficients of the Pauli decomposition. If the number of
        coefficients is one greater than the number of entries in
        ``re_ope``, the first coefficient is treated as the identity
        contribution and excluded from the shot allocation.
    re_ope : sequence of str
        Compact Pauli strings corresponding to the measured Hamiltonian
        terms.
    state : QuantumState
        Quantum state used to calculate the Pauli variances.
    n_qubits : int
        Total number of qubits.
    total_shots : int
        Total measurement-shot budget.
    min_shots_nonzero : int, optional
        Minimum number of shots assigned to each term with nonzero
        allocation weight. The default is 1.
    round_mode : {"real", "floor", "ceil", "round", "int"}, optional
        Method used to convert continuous allocations to shot counts.
        ``"real"`` returns the unrounded allocation. The default is
        ``"int"``, which initially applies ``floor``.
    enforce_total : bool, optional
        If True, adjust integer allocations so that their sum matches
        ``total_shots`` whenever permitted by the minimum-shot
        constraint. The default is True.

    Returns
    -------
    shots : numpy.ndarray
        Number of shots allocated to each measured Pauli term.
    variances : numpy.ndarray
        Measurement variances of the Pauli terms.
    weights : numpy.ndarray
        Allocation weights ``|c_i| * sqrt(Var[P_i])``.

    Raises
    ------
    ValueError
        If the number of coefficients is inconsistent with the number of
        Pauli terms or if an unsupported ``round_mode`` is specified.
    """
    re_num_arr = np.asarray(re_num, dtype=float).ravel()
    ops = list(re_ope)

    if len(re_num_arr) == len(ops):
        coeffs = re_num_arr
    elif len(re_num_arr) == len(ops) + 1:
        coeffs = re_num_arr[1:]
    else:
        raise ValueError(
            f"Length mismatch: {len(re_num_arr)} coefficients for {len(ops)} Pauli terms."
        )

    variances = np.array(
        [estimate_pauli_variance_from_state(op, n_qubits, state) for op in ops],
        dtype=float
    )
    weights = np.abs(coeffs) * np.sqrt(variances)

    delta_w = 1e-12  
    weights[weights < delta_w] = 0.0
    active = weights > 0.0
    if not np.any(active):
        return np.zeros(len(ops)), variances, weights

    wsum = weights[active].sum()
    raw = np.zeros(len(ops), dtype=float)
    raw[active] = total_shots * (weights[active] / wsum)

    if round_mode == "real":
        return raw, variances, weights

    if round_mode == "ceil":
        shots = np.ceil(raw).astype(int)
    elif round_mode == "floor":
        shots = np.floor(raw).astype(int)
    elif round_mode == "round":
        shots = np.rint(raw).astype(int)
    elif round_mode == "int":
        shots = np.floor(raw).astype(int)
    else:
        raise ValueError(f"Unknown round_mode={round_mode}")

    if min_shots_nonzero and min_shots_nonzero > 0:
        for i in range(len(shots)):
            if active[i] and shots[i] < min_shots_nonzero:
                shots[i] = min_shots_nonzero

    if enforce_total:
        diff = int(total_shots - shots.sum())
        if diff != 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac) if diff > 0 else np.argsort(frac)
            for i in order:
                if not active[i]:
                    continue
                if diff == 0:
                    break
                if diff > 0:
                    shots[i] += 1
                    diff -= 1
                else:
                    floor_i = min_shots_nonzero if min_shots_nonzero else 0
                    if shots[i] > floor_i:
                        shots[i] -= 1
                        diff += 1

    return shots, variances, weights
