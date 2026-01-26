"""
This module provides utilities for:
- Parsing compact Pauli strings (e.g., 'X0X1Y2Y3') into Qulacs-compatible observables.
- Applying appropriate basis transformations (X → H, Y → Rz(-π/2)+H) for Z-basis measurement.
- Efficiently sampling ±1 values from quantum states with vectorized bitwise operations.
- Estimating expectation values from sampling results and comparing them to exact theoretical values.

Main entry point:
    - `expect_sampling(term, shots, n_qubits, base_state)`:
        Returns both theoretical and sampling-based expectation values for a given Pauli term.

Compatible with Python 3.7+ and Qulacs 0.5.0+.
"""

import re
import numpy as np
from qulacs import QuantumCircuit, Observable
from qulacs.gate import H, RZ
from typing import Dict, Tuple

_BASIS_CIRCUIT_CACHE: Dict[Tuple[Tuple[Tuple[int, str], ...], int], QuantumCircuit] = {}

def _basis_circuit_for_term(pauli_ops: Dict[int, str], n_qubits: int) -> QuantumCircuit:
    """
    Constructs and caches the basis change circuit for a given Pauli term.

    For each qubit:
        - 'X': Apply Hadamard (H)
        - 'Y': Apply Rz(-π/2) followed by H
        - 'Z' or 'I': No change

    Parameters:
        pauli_ops (Dict[int, str]): Dictionary mapping qubit indices to Pauli letters.
        n_qubits (int): Total number of qubits in the system.

    Returns:
        QuantumCircuit: Basis transformation circuit cached by (term, n_qubits).
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
    Formats a Pauli term dictionary into Qulacs Observable string syntax.

    Example:
        {0:'X', 2:'Y'} → "X 0 Y 2"

    Parameters:
        pauli_ops (dict): Mapping from qubit indices to Pauli operators.

    Returns:
        str: Qulacs-formatted operator string.
    """
    if not pauli_ops:
        return ""  
    parts = []
    for q in sorted(pauli_ops.keys()):
        parts.append(f"{pauli_ops[q]} {q}")
    return " ".join(parts)

def _parse_pauli_term_compact(term: str) -> Dict[int, str]:
    """
    Parses compact Pauli strings like 'X0X1Y2' into a dictionary format.

    Example:
        'X0Y2' → {0:'X', 2:'Y'}

    An empty string ('') is interpreted as the identity operator.

    Parameters:
        term (str): Compact Pauli string.

    Returns:
        Dict[int, str]: Mapping from qubit indices to Pauli letters.
    """
    ops = {}
    if not term:  
        return ops
    for pauli, idx in re.findall(r'([XYZ])(\d+)', term):
        ops[int(idx)] = pauli
    return ops

def _samples_to_pm1(samples, support_qubits, n_qubits):
    """
    Converts integer measurement samples to ±1 values for a given Z-term.

    For each sample, computes the parity of selected qubits:
        - Even parity → +1
        - Odd parity → -1

    Parameters:
        samples (np.ndarray): Array of measurement results (integers).
        support_qubits (List[int]): Qubit indices involved in the Pauli term.
        n_qubits (int): Total number of qubits in the system.

    Returns:
        np.ndarray: Array of ±1 values of shape (len(samples),).
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
    print("support:", support_qubits, " supp dtype:", np.asarray([int(q) for q in support_qubits]).dtype)
    return (1 - 2 * parity).astype(np.int8)

def batched_sampling_est_prob(qu_state, sub, shots_per_trial, n_trials,
                              max_batch_shots=2_000_000):
    """
    Performs repeated sampling in batches and returns estimated probabilities.

    This function divides `n_trials` into batches and collects sampling data
    to estimate the probability of measuring all zeros on `sub` qubits.

    Parameters:
        qu_state (QuantumState): The quantum state to sample from.
        sub (int): Number of qubits to check for |0⟩ in sampling.
        shots_per_trial (int): Number of shots per sampling trial.
        n_trials (int): Total number of trials to perform.
        max_batch_shots (int): Upper bound on total shots per batch.

    Returns:
        np.ndarray: Estimated probabilities (length = `n_trials`) of all-zero outcomes.
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
    Performs basis transformation and estimates expectation value by sampling.

    Parameters:
        term (str): Pauli term in compact form (e.g., 'X0Y1').
        shots (int): Number of measurement samples to draw.
        n_qubits (int): Total number of qubits.
        base_state (QuantumState): The input quantum state to measure.

    Returns:
        Tuple[float, float, float, float]:
            (samp_pro, samp_exp, theo_pro, theo_expect)
        - samp_pro   : Sampling-based probability of +1 outcome
        - samp_exp   : Sampling-based expectation value (2*p - 1)
        - theo_pro   : Theoretical probability of +1 outcome
        - theo_expect: Theoretical expectation value using Observable
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
