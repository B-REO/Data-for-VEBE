"""
This module provides an implemantation ansatz proposed by Gard et al. 
on Qulacs quantum circuits.

Functions included:
- add_ansatz_unit: Add a two-qubit ansatz unit acting on neighboring qubits.
- add_ansatz_layer: Add one ansatz layer composed of multiple ansatz units.
- add_ansatz: Construct the complete ansatz circuit.

Use cases:
- Preparing variational quantum states for VQE.
- Preparing variational quantum states for VEBE.


Requirements:
- Python 3.7+
- numpy
- qulacs

Note:
- This module only constructs ansatz circuits.
  Optimization procedures and cost-function evaluations are implemented
  separately.

Reference:
Gard et al.,
“Efficient symmetry-preserving state preparation circuits for the
variational quantum eigensolver algorithm”,
npj Quantum Information 6, 10 (2020).
"""

import math
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import X, CNOT, RY, RZ

def add_ansatz_unit(
    circuit: QuantumCircuit,
    qubit: int,
    phi: float,
    theta: float,
) -> None:
    """Add one two-qubit entangling ansatz unit.

    This unit acts on neighboring qubits `qubit` and `qubit + 1`.
    """
    q0 = qubit
    q1 = qubit + 1

    circuit.add_gate(CNOT(q1, q0))
    circuit.add_gate(RZ(q1, -phi - np.pi))
    circuit.add_gate(RY(q1, -theta - np.pi / 2))
    circuit.add_gate(CNOT(q0, q1))
    circuit.add_gate(RZ(q1, phi + np.pi))
    circuit.add_gate(RY(q1, theta + np.pi / 2))
    circuit.add_gate(CNOT(q1, q0))

def add_ansatz_layer(
    circuit: QuantumCircuit,
    n_orbitals: int,
    theta_list,
    phi_list,
    offset: int,
    n_ancilla: int = 0,
) -> None:
    """Add one layer of nearest-neighbor ansatz units."""
    half = n_orbitals // 2

    for i in range(half):
        add_ansatz_unit(
            circuit,
            n_ancilla + 2 * i,
            phi_list[offset + i],
            theta_list[offset + i],
        )

    for i in range(half - 1):
        add_ansatz_unit(
            circuit,
            n_ancilla + 2 * i + 1,
            phi_list[offset + half + i],
            theta_list[offset + half + i],
        )
        
def add_ansatz(
    circuit: QuantumCircuit,
    theta_list,
    phi_list,
    n_orbitals: int,
    n_electrons: int,
    n_ancilla: int = 0,
) -> QuantumCircuit:
    """Construct the Jastrow-type ansatz circuit.

    Parameters
    ----------
    circuit:
        Qulacs quantum circuit to which gates are added.
    theta_list, phi_list:
        Variational parameters.
    n_orbitals:
        Number of spin orbitals / qubits used by this ansatz block.
    n_electrons:
        Number of initially occupied orbitals.
    n_ancilla:
        Number of ancilla qubits for the Block-Encoding algorithm.
        If not using the Block-Encoding method, please ignore or input 0.

    Returns
    -------
    QuantumCircuit
        The input circuit with the ansatz appended.
    """
    if n_electrons > n_orbitals:
        raise ValueError("n_electrons must be <= n_orbitals.")

    if n_orbitals < 2:
        raise ValueError("n_orbitals must be at least 2.")

    # Prepare Hartree-Fock state as initial occupation.
    for i in range(n_electrons):
        circuit.add_gate(X(n_ancilla + i))

    # Number of orbital-pair combinations: nCk.
    n_combinations = math.comb(n_orbitals, n_electrons)

    # Each layer consumes n_orbitals - 1 parameters.
    n_layers = n_combinations // (n_orbitals - 1)

    required_params = n_layers * (n_orbitals - 1)

    if len(theta_list) < required_params or len(phi_list) < required_params:
        raise ValueError(
            f"theta_list and phi_list must each contain at least "
            f"{required_params} parameters."
        )

    for layer in range(n_layers):
        offset = layer * (n_orbitals - 1)
        add_ansatz_layer(

            circuit,
            n_orbitals,
            theta_list,
            phi_list,
            offset,
            n_ancilla,
        )

    return circuit
