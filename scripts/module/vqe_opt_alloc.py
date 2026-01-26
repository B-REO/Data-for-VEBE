"""
This module provides functions to estimate variances of Pauli-string measurements
from a given quantum state and to compute an optimal allocation of measurement shots
across Pauli terms under a fixed total shot budget.

Key features:
- Conversion from compact Pauli strings (e.g. 'X0Y1Z2') to Qulacs Observable format
- Estimation of Pauli measurement variances using exact expectation values
- Shot allocation proportional to |c_i| * sqrt(Var[P_i]), following standard VQE theory
- Flexible rounding strategies and optional enforcement of total shot count

Typical use case:
- Variance-aware measurement allocation in VQE or related variational algorithms
- Post-processing of Hamiltonian terms represented as Pauli strings

This module is designed to work with Qulacs QuantumState objects and assumes
that expectation values can be computed exactly on a simulator.
"""

import numpy as np
import re
from qulacs import QuantumState, Observable

def _compact_to_qulacs_pauli_string(term: str) -> str:
    """
    Convert a compact Pauli string into Qulacs Observable string format.

    Example:
        'X0X1Y2Y3' -> 'X 0 X 1 Y 2 Y 3'
        ''         -> '' (identity)

    Parameters:
        term (str): Compact Pauli string representation.

    Returns:
        str: Pauli operator string formatted for Qulacs Observable.
    """
    if not term:
        return ""
    parts = []
    for p, idx in re.findall(r'([XYZ])(\d+)', term):
        parts.append(f"{p} {idx}")
    return " ".join(parts)

def estimate_pauli_variance_from_state(term: str, n_qubits: int, state: QuantumState) -> float:
    """
    Estimate the measurement variance of a Pauli string from a quantum state.

    For a Pauli operator P, the measurement outcomes are ±1 after basis rotation.
    The variance is computed as:
        Var(P) = 1 - <P>^2

    For the identity operator (term == ''), <I> = 1 and Var = 0.

    Parameters:
        term (str): Compact Pauli string (e.g. 'X0Y1Z2'). Empty string denotes identity.
        n_qubits (int): Total number of qubits.
        state (QuantumState): Quantum state used to compute the expectation value.

    Returns:
        float: Estimated variance of the Pauli measurement outcome.
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
    Compute an optimal measurement shot allocation for VQE Pauli terms.

    The allocation is based on the standard variance-weighted rule:
        shots_i ∝ |c_i| * sqrt(Var[P_i])

    where c_i is the coefficient of the Pauli term and Var[P_i] is the measurement variance.

    Parameters:
        re_num (array-like):
            Coefficients of Pauli terms. If length is len(re_ope)+1, the first element
            is assumed to be a constant shift and ignored.
        re_ope (list of str):
            Compact Pauli strings corresponding to Hamiltonian terms.
        state (QuantumState):
            Quantum state used to estimate Pauli variances.
        n_qubits (int):
            Total number of qubits.
        total_shots (int):
            Total measurement budget to be allocated.
        min_shots_nonzero (int, optional):
            Minimum number of shots assigned to nonzero-weight terms.
        round_mode (str, optional):
            Rounding mode for shot counts:
            'real', 'floor', 'ceil', 'round', or 'int'.
        enforce_total (bool, optional):
            If True, adjusts allocations to exactly match total_shots.

    Returns:
        shots (np.ndarray):
            Allocated number of shots per Pauli term.
        variances (np.ndarray):
            Estimated variances for each Pauli term.
        weights (np.ndarray):
            Weight values |c_i| * sqrt(Var[P_i]) used for allocation.
    """
    re_num_arr = np.asarray(re_num, dtype=float).ravel()
    ops = list(re_ope)

    if len(re_num_arr) == len(ops):
        coeffs = re_num_arr
    elif len(re_num_arr) == len(ops) + 1:
        coeffs = re_num_arr[1:]
    else:
        L = min(len(re_num_arr), len(ops))
        coeffs = re_num_arr[:L]
        ops = ops[:L]

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
