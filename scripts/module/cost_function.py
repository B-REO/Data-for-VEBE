"""
Cost functions for VEBE and VQE variational optimizations.

This module provides objective functions for ground-state optimization
using the symmetry-preserving ansatz proposed by Gard et al. The quantum
circuits are implemented with Qulacs.

Functions
---------
cost_vebe
    Evaluate the VEBE objective from the block-encoding success
    probability.

cost_vqe
    Evaluate the VQE objective as the expectation value of the
    Hamiltonian.

Notes
-----
This module evaluates objective functions only. Classical optimization
procedures such as BFGS, COBYLA, or SLSQP are implemented separately in
the execution scripts.

The VEBE implementation requires the PItBE package included under
``vendor/PItBE``.

References
----------
Gard, B. T. et al., "Efficient symmetry-preserving state preparation
circuits for the variational quantum eigensolver algorithm,"
npj Quantum Information 6, 10 (2020).
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
    """
    Evaluate the VEBE objective function.

    The variational ansatz is applied to the main register, followed by
    the block-encoding circuit. The objective is obtained from the
    probability of measuring all ancilla qubits in the zero state.

    Parameters
    ----------
    variable : list
        Variational parameters. The first ``comb(n_main, n_electrons)``
        elements are used as theta parameters and the remaining elements
        as phi parameters.
    n_main : int
        Number of qubits in the main register.
    n_ancilla : int
        Number of ancilla qubits used for block encoding.
    n_electrons : int
        Number of initially occupied spin orbitals.
    normalization_factor : float
        Normalization factor of the block-encoded Hamiltonian.
    gate_b
        Gate preparing the ancilla amplitudes associated with the
        absolute values of the normalized Hamiltonian coefficients.
    sign_gate
        Gate encoding the signs of the normalized Hamiltonian
        coefficients.
    control_states : list
        Control states associated with the conditionally applied Pauli
        operators.
    operator_list : list
        Pauli operators used to construct the block-encoded Hamiltonian.

    Returns
    -------
    float
        VEBE objective value,
        ``-abs(normalization_factor) * sqrt(p_succ)``, where ``p_succ``
        is the probability of measuring all ancilla qubits in the zero
        state.
    """
    # Prepare the initial quantum state and quantum circuit.
    state = QuantumState(n_main + n_ancilla)
    circuit = QuantumCircuit(n_main + n_ancilla)

    # Split the optimization variables into theta and phi parameters.
    n_ansatz_params = math.comb(n_main, n_electrons)
    theta_list = variable[:n_ansatz_params]
    phi_list = variable[n_ansatz_params:]

    # Add the ansatz circuit to the main register.
    add_ansatz(
        circuit,
        theta_list,
        phi_list,
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
    """
    Evaluate the VQE objective function.

    The function prepares the variational state using the
    symmetry-preserving ansatz and returns the expectation value of the
    supplied Hamiltonian.

    Parameters
    ----------
    variable : list
        Variational parameters. The first ``comb(n_main, n_electrons)``
        elements are used as theta parameters and the remaining elements
        as phi parameters.
    n_main : int
        Number of qubits in the main register.
    n_electrons : int
        Number of initially occupied spin orbitals.
    hamiltonian
        Qulacs observable representing the Hamiltonian.

    Returns
    -------
    float
        Expectation value of the Hamiltonian for the variational state.
    """
    # Prepare the initial quantum state and quantum circuit.
    state = QuantumState(n_main)
    circuit = QuantumCircuit(n_main)

    # Split the optimization variables into theta and phi parameters.
    n_ansatz_params = math.comb(n_main, n_electrons)
    theta_lst = variable[:n_ansatz_params]
    phi_lst = variable[n_ansatz_params:]

    # Add the ansatz circuit to the main register.
    add_ansatz(
        circuit,
        theta_lst,
        phi_lst,
        n_main,
        n_electrons,
    )

    # Apply the ansatz circuit to the quantum state.
    circuit.update_quantum_state(state)

    # Return the expectation value as the objective value.
    return hamiltonian.get_expectation_value(state)
