"""
This module provides objective functions for variational quantum
algorithms based on the ansatz proposed by Gard et al. 
[https://www.nature.com/articles/s41534-019-0240-1.pdf] 
and Qulacs quantum circuits.

Functions included:
- cost_vebe: Cost function used in VEBE optimization.
- cost_vqe: Cost function used in VEBE optimization.

Use cases:
- Ground-state energy calculations using VQE.
- Ground-state energy calculations using VEBE.
- Benchmark comparisons between VQE and VEBE.


Requirements:
- Python 3.7+
- numpy
- qulacs

Note:
- This module only evaluates objective functions.
- Optimization procedures (e.g., BFGS, COBYLA, SLSQP) should be implemented separately in the execution scripts.
"""

import math
import numpy as np

from qulacs import QuantumState
from qulacs import QuantumCircuit
try:
    import pitbe
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pitbe could not be imported. "
        "Please clone vendor/PItBE and add it to PYTHONPATH."
    ) from exc

from module.ansatz import add_ansatz

def cost_vebe(
    variable: list,
    n_main: int,
    n_ancilla: int,
    n_electrons: int,
    normalization_factor: float,
    gate_b,
    sign_gate,
    control_states: list,
    operator_list: list,
) -> float:
    """Evaluate the VEBE cost function.

    Parameters
    ----------
    variable : list
        Variational parameters to be optimized. The first half is used as
        phi parameters, and the second half is used as theta parameters.
    n_main : int
        Number of main qubits used to represent the quantum state.
    n_ancilla : int
        Number of ancilla qubits used for the block-encoding circuit.
    n_electrons : int
        Number of initially occupied orbitals.
    normalization_factor : float
        Normalization factor of the block-encoded Hamiltonian.
    gate_b : QuantumGate
        Quantum gate that prepares the amplitudes corresponding to the
        absolute values of the normalized coefficients on the ancilla qubits.
    sign_gate : QuantumGate
        Quantum gate that encodes the signs of the normalized coefficients.
    control_states : list
        List of control states used to apply each Pauli operator conditionally.
    operator_list : list
        List of Pauli operators applied in the block-encoding circuit.

    Returns
    -------
    float
        Objective value for VEBE optimization, given by
        -normalization_factor * sqrt(p_succ).
    """
    # Prepare the initial quantum state and quantum circuit.
    state = QuantumState(n_main + n_ancilla)
    circuit = QuantumCircuit(n_main + n_ancilla)

    # Split the optimization variables into phi and theta parameters.
    n_ansatz_params = math.comb(n_main, n_electrons)
    phi_list = variable[:n_ansatz_params]
    theta_list = variable[n_ansatz_params:]

    # Add the ansatz circuit to the main register.
    add_ansatz(
        circuit,
        phi_list,
        theta_list,
        n_main,
        n_electrons,
        n_ancilla,
    )

    # Apply the block-encoding circuit for the Hamiltonian.
    circuit.add_gate(gate_b)
    circuit.add_gate(sign_gate)

    for operator, control_state in zip(operator_list, control_states):
        pitbe.circ_make(
            operator,
            control_state,
            circuit,
            n_main + n_ancilla,
            n_ancilla,
        )

    circuit.add_gate(gate_b.get_inverse())
    circuit.update_quantum_state(state)

    # Compute the theoretical success probability of measuring
    # all ancilla qubits in |0>.
    marginal_order = [0] * n_ancilla + [2] * n_main
    success_probability = state.get_marginal_probability(marginal_order)

    # Return the objective value for minimization.
    return -np.sqrt(abs(success_probability)) * abs(normalization_factor)

def cost_vqe(
    variable: list, 
    n_main: int, 
    n_electrons: int,
    hamiltonian
) -> float:
    """Evaluate the VQE cost function.
    Parameters
    ----------
    variable : list
        Variational parameters to be optimized. The first half is used as
        phi parameters, and the second half is used as theta parameters.
    n_main : int
        Number of main qubits used to represent the quantum state.
    n_electrons : int
        Number of initially occupied orbitals.
    hamiltonian : 
        Hamiltonian for which the expectation value is evaluated.
        Qulacs Observable used to evaluate the expectation value.

    Returns
    -------
    float
        Objective value for VQE optimization.
    """
    # Prepare the initial quantum state and quantum circuit.
    state = QuantumState(n_main) 
    circuit = QuantumCircuit(n_main)

    # Split the optimization variables into phi and theta parameters.
    n_ansatz_params = math.comb(n_main, n_electrons)
    phi_lst = variable[:n_ansatz_params]
    theta_lst = variable[n_ansatz_params:]

    # Add the ansatz circuit to the main register.
    add_ansatz(
        circuit,
        phi_lst, 
        theta_lst, 
        n_main, 
        n_electrons
        )

    # Apply the ansatz circuit to the quantum state
    circuit.update_quantum_state(state) 

    # Return the expectation value as the objective value.
    return hamiltonian.get_expectation_value(state)
