"""
Sampling utilities for Pauli measurements and VEBE success probabilities.

This module provides functions for parsing compact Pauli strings,
constructing and caching measurement-basis transformation circuits,
converting computational-basis samples to Pauli measurement outcomes,
and performing repeated sampling for VEBE and VQE simulations.

Functions
---------
batched_sampling_est_prob
    Estimate block-encoding success probabilities over repeated sampling
    trials using batched execution.
    
expect_sampling
    Estimate the expectation value of a Pauli operator from measurement
    samples and return the corresponding exact simulator value.

Notes
-----
Compact Pauli strings use representations such as ``"X0Y1Z2"``.
An empty string represents the identity operator.
"""

import re
import numpy as np
from qulacs import QuantumCircuit, Observable
from qulacs.gate import H, RZ
from typing import Dict, Tuple

_BASIS_CIRCUIT_CACHE: Dict[Tuple[Tuple[Tuple[int, str], ...], int], QuantumCircuit] = {}

def _basis_circuit_for_term(pauli_ops: Dict[int, str], n_qubits: int) -> QuantumCircuit:
    """
    Construct and cache the basis-change circuit for a Pauli term.

    X measurements are transformed to the computational basis with a
    Hadamard gate. Y measurements use an ``RZ(-pi/2)`` rotation followed
    by a Hadamard gate. Z measurements require no basis transformation.

    Parameters
    ----------
    pauli_ops : dict of int to str
        Mapping from qubit indices to Pauli operators.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    QuantumCircuit
        Basis-change circuit. Circuits are cached using the Pauli term
        and total number of qubits as the cache key.
    """
    key = (tuple(sorted(pauli_ops.items())), n_qubits)
    if key in _BASIS_CIRCUIT_CACHE:
        return _BASIS_CIRCUIT_CACHE[key]
    circ = QuantumCircuit(n_qubits)
    for q, a in pauli_ops.items():
        if a == 'X':
            circ.add_gate(H(q))
        elif a == 'Y':
            circ.add_gate(RZ(q, -np.pi/2.0))
            circ.add_gate(H(q))
    _BASIS_CIRCUIT_CACHE[key] = circ
    return circ

def _format_qulacs_op_string(pauli_ops: dict) -> str:
    """
    Convert a Pauli-operator dictionary to Qulacs observable syntax.

    Parameters
    ----------
    pauli_ops : dict
        Mapping from qubit indices to Pauli operators.

    Returns
    -------
    str
        Qulacs-formatted Pauli string. For example,
        ``{0: "X", 2: "Y"}`` is converted to ``"X 0 Y 2"``.
        An empty dictionary returns an empty string.
    """
    if not pauli_ops:
        return ""  
    parts = []
    for q in sorted(pauli_ops.keys()):
        parts.append(f"{pauli_ops[q]} {q}")
    return " ".join(parts)

def _parse_pauli_term_compact(term: str) -> Dict[int, str]:
    """
    Parse a compact Pauli string into a qubit-operator mapping.

    Parameters
    ----------
    term : str
        Compact Pauli string such as ``"X0Y2"``. An empty string
        represents the identity operator.

    Returns
    -------
    dict of int to str
        Mapping from qubit indices to Pauli operators. For example,
        ``"X0Y2"`` is converted to ``{0: "X", 2: "Y"}``.
    """
    ops = {}
    if not term:  
        return ops
    for pauli, idx in re.findall(r'([XYZ])(\d+)', term):
        ops[int(idx)] = pauli
    return ops

def _samples_to_pm1(samples, support_qubits, n_qubits):
    """
    Convert computational-basis samples to Pauli outcomes of +1 or -1.

    The parity of the measured bits on ``support_qubits`` determines the
    corresponding Pauli measurement outcome. Even parity gives +1 and
    odd parity gives -1.

    Parameters
    ----------
    samples : array_like
        Integer representations of computational-basis measurement
        outcomes.
    support_qubits : sequence of int
        Qubit indices on which the Pauli operator acts nontrivially.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    numpy.ndarray
        Array of +1 and -1 measurement outcomes with one element per
        input sample.
    """
    if samples is None:
        return np.array([], dtype=np.int8)
    M = len(samples)
    if M == 0:
        return np.array([], dtype=np.int8)
    if not support_qubits:
        return np.ones(M, dtype=np.int8)

    if n_qubits <= 63:
        s = np.asarray(samples, dtype=np.uint64)
        supp = np.asarray([int(q) for q in support_qubits], dtype=np.int64)

        try:
            bits = ((s[:, None] >> supp[None, :]) & 1).astype(np.uint8)
        except Exception:
            cols = [((s >> int(q)) & 1).astype(np.uint8) for q in supp]
            bits = np.stack(cols, axis=1) if cols else np.zeros((M, 0), dtype=np.uint8)

        parity = np.bitwise_xor.reduce(bits, axis=1) if bits.shape[1] else np.zeros(M, dtype=np.uint8)
        return (1 - 2 * parity).astype(np.int8)

    s_obj = np.asarray(samples, dtype=object)
    supp_list = [int(q) for q in support_qubits]

    def parity_of(val):
        p = 0
        for q in supp_list:
            p ^= ((val >> q) & 1)
        return p

    parity = np.fromiter((parity_of(v) for v in s_obj), dtype=np.uint8, count=M)
    return (1 - 2 * parity).astype(np.int8)

def batched_sampling_est_prob(qu_state, sub, shots_per_trial, n_trials,
                              max_batch_shots=2_000_000):
    """
    Estimate all-zero probabilities over repeated sampling trials.

    Sampling is performed in batches to limit the number of shots passed
    to Qulacs at once. For each trial, the function estimates the
    probability that the lowest ``sub`` qubits are all measured in the
    zero state.

    Parameters
    ----------
    qu_state
        Qulacs quantum state to sample.
    sub : int
        Number of lowest-index qubits required to be in the zero state.
    shots_per_trial : int
        Number of measurement shots in each trial.
    n_trials : int
        Number of independent sampling trials.
    max_batch_shots : int, optional
        Approximate maximum number of shots sampled in one batch. The
        default is 2,000,000.

    Returns
    -------
    numpy.ndarray
        Estimated all-zero probabilities for the individual trials.
        The returned array has length ``n_trials``.

    Notes
    -----
    If ``sub <= 0``, the probability is defined as one for every trial.
    """
    shots_per_trial = int(shots_per_trial)
    n_trials = int(n_trials)
    if sub <= 0:
        return np.ones(n_trials, dtype=float)

    trials_per_batch = max(1, max_batch_shots // max(1, shots_per_trial))

    mask = (1 << sub) - 1
    out = np.empty(n_trials, dtype=float)
    w = 0
    while w < n_trials:
        curr = min(trials_per_batch, n_trials - w)
        total = shots_per_trial * curr
        if total > 2_000_000_000:
            curr = max(1, 2_000_000_000 // max(1, shots_per_trial))
            total = shots_per_trial * curr

        samples = np.asarray(qu_state.sampling(int(total)), dtype=np.uint64)
        good = (samples & mask) == 0
        good = good.reshape(curr, shots_per_trial)
        out[w:w+curr] = good.mean(axis=1)
        w += curr

    return out

def expect_sampling(term, shots, n_qubits, base_state):
    """
    Estimate the expectation value of a Pauli term by sampling.

    The required basis transformation is applied to a copy of
    ``base_state`` before computational-basis sampling. The function also
    evaluates the exact expectation value using a Qulacs ``Observable``.

    Parameters
    ----------
    term : str
        Pauli term in compact notation, such as ``"X0Y1"``. An empty
        string represents the identity operator.
    shots : int
        Number of measurement shots.
    n_qubits : int
        Total number of qubits.
    base_state
        Qulacs quantum state to be measured. The supplied state is not
        modified.

    Returns
    -------
    samp_pro : float
        Sampling-based probability of obtaining the +1 eigenvalue.
    samp_exp : float
        Sampling-based expectation value.
    theo_pro : float
        Exact probability of the +1 eigenvalue calculated from the
        simulator expectation value.
    theo_expect : float
        Exact expectation value calculated using a Qulacs ``Observable``.

    Notes
    -----
    If ``shots <= 0``, the exact simulator expectation value is returned
    in place of a sampled estimate. This is used for terms, such as the
    identity operator, that do not require sampling.
    """
    shots = int(shots)
    pauli_ops = _parse_pauli_term_compact(term)

    if pauli_ops:
        obs = Observable(n_qubits)
        op_str = _format_qulacs_op_string(pauli_ops)
        obs.add_operator(1.0, op_str)
        samp_state_for_theory = base_state.copy()
        theo_expect = float(obs.get_expectation_value(samp_state_for_theory))
    else:
        theo_expect = 1.0

    if shots <= 0:
        samp_exp = theo_expect
        samp_pro = (samp_exp + 1.0) / 2.0
        theo_pro = (theo_expect + 1.0) / 2.0
        return float(samp_pro), float(samp_exp), float(theo_pro), float(theo_expect)

    work = base_state.copy()
    circ = _basis_circuit_for_term(pauli_ops, n_qubits)
    circ.update_quantum_state(work)

    shots = int(shots)
    raw = work.sampling(shots)                      
    samples = np.asarray(raw, dtype=np.uint64)   
    support = sorted(pauli_ops.keys())
    pm1 = _samples_to_pm1(samples, support, n_qubits)             
    
    samp_exp = float(pm1.mean()) 
    samp_pro = 0.5 * (samp_exp + 1.0)      
    theo_pro = float(0.5 * (theo_expect + 1.0))   

    return samp_pro, samp_exp, theo_pro, theo_expect
