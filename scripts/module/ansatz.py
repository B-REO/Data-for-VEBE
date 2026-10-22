"""
Implementation of the symmetry-preserving ansatz proposed by Gard et al.

This module provides functions for constructing the particle-number-
preserving variational quantum circuit used in the VQE and VEBE
calculations in this repository. The circuit is implemented using Qulacs.

Functions
---------
add_ansatz_unit
    Add a two-qubit ansatz unit acting on neighboring qubits.

add_ansatz_layer
    Add one layer of nearest-neighbor ansatz units.

add_ansatz
    Prepare the initial occupation state and append the complete ansatz
    circuit.

Notes
-----
The ansatz acts on the main register. When it is used as part of a VEBE
circuit, ``n_ancilla`` specifies the offset of the main register.

References
----------
Gard, B. T. et al., "Efficient symmetry-preserving state preparation
circuits for the variational quantum eigensolver algorithm,"
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
    """
    Add one two-qubit symmetry-preserving ansatz unit.

    The unit acts on neighboring qubits ``qubit`` and ``qubit + 1`` and
    consists of CNOT and single-qubit rotation gates.

    Parameters
    ----------
    circuit : QuantumCircuit
        Qulacs quantum circuit to which the ansatz unit is appended.
    qubit : int
        Index of the first qubit of the neighboring pair.
    phi : float
        Variational rotation parameter.
    theta : float
        Variational rotation parameter.

    Returns
    -------
    None
        The supplied circuit is modified in place.
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
    """
    Add one layer of nearest-neighbor ansatz units.

    The layer first applies ansatz units to even neighboring pairs and
    then to odd neighboring pairs, resulting in ``n_orbitals - 1``
    two-qubit units per layer.

    Parameters
    ----------
    circuit : QuantumCircuit
        Qulacs quantum circuit to which the layer is appended.
    n_orbitals : int
        Number of qubits in the main register.
    theta_list : array_like
        Sequence of theta parameters.
    phi_list : array_like
        Sequence of phi parameters.
    offset : int
        Starting index in ``theta_list`` and ``phi_list`` for this layer.
    n_ancilla : int, optional
        Number of ancilla qubits preceding the main register.

    Returns
    -------
    None
        The supplied circuit is modified in place.
    """
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
    """
    Construct the symmetry-preserving ansatz proposed by Gard et al.

    The function first prepares the initial occupation state by applying
    X gates to the first ``n_electrons`` qubits of the main register.
    It then appends layers of nearest-neighbor ansatz units.

    Parameters
    ----------
    circuit : QuantumCircuit
        Qulacs quantum circuit to which the ansatz is appended.
    theta_list : array_like
        Sequence of theta variational parameters.
    phi_list : array_like
        Sequence of phi variational parameters.
    n_orbitals : int
        Number of spin orbitals represented by the main register.
    n_electrons : int
        Number of initially occupied spin orbitals.
    n_ancilla : int, optional
        Number of ancilla qubits preceding the main register. The default
        is 0.

    Returns
    -------
    QuantumCircuit
        The input circuit with the initial occupation state and ansatz
        layers appended.

    Raises
    ------
    ValueError
        If ``n_electrons`` exceeds ``n_orbitals``, if fewer than two
        orbitals are specified, or if insufficient variational parameters
        are supplied.
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
