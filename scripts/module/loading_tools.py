"""
Utility functions for loading and organizing simulation data.

This module provides helper functions for locating RHF input files,
loading simulation results, extracting metadata from file and directory
names, constructing qubit Hamiltonians, and preparing quantum states for
VEBE and VQE simulations.

The functions are primarily used by the analysis and optimization
notebooks associated with this repository. File and directory names are
assumed to follow the naming conventions used by the simulation scripts.

Functions
---------
find_rhf_files_with_metadata
    Locate CASCI-based RHF files and extract molecular metadata.

find_rhf_files_with_metadata_for_fci
    Locate FCI-based RHF files and extract molecular metadata.

get_folder_paths_and_values
    Locate numerically named directories and return their numeric values.

get_statenpz_paths_and_values
    Locate NPZ files and extract numerical parameters from their filenames.

get_txt_paths_and_values
    Locate TXT files and extract numerical parameters from their filenames.

load_state_on_main
    Embed a state vector into the main register of a larger quantum state.

load_target_and_shift
    Load the target energy and Hamiltonian shift for an optimization run.

load_trajectories
    Load optimization trajectories from multiple random-seed runs.

qubit_hamiltonian
    Construct a Jordan-Wigner-transformed qubit Hamiltonian from an RHF file.

read_value
    Read a numerical value associated with a specified key from a TXT file.

seed_folders
    Locate random-seed directories used for repeated optimization runs.

Notes
-----
These utilities assume the directory structure and naming conventions
used by the simulation scripts in this repository.
"""

import numpy as np
import os
import re

def find_rhf_files_with_metadata(folder_path):
    """
    Locate CASCI-based RHF files and extract metadata from their filenames.

    The directory tree rooted at ``folder_path`` is searched recursively.
    Files are expected to follow the naming convention

    ``casci_<system>_..._<config>_<integer>_<decimal>.rhf``,

    where ``<config>`` represents the active-space specification and the
    final two integer fields define a floating-point parameter such as
    the bond length.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Root directory to search recursively.

    Returns
    -------
    list of tuple
        Tuples of the form
        ``(file_path, system_name, config_string, value)``, sorted in
        ascending order of ``value``.

    Notes
    -----
    Files that do not match the expected naming convention are skipped.
    If ``folder_path`` does not exist, an empty list is returned.
    """
    matched_files = []

    if not os.path.isdir(folder_path):
        print(f"Not found: {folder_path}")
        return []

    pattern = re.compile(
        r'casci_([A-Za-z0-9]+).*?_([0-9]+_[0-9]+)_([0-9]+)_([0-9]+)\.rhf$', re.IGNORECASE)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".rhf"):
                match = pattern.search(file)
                if match:
                    system_name = match.group(1)          
                    config_string = match.group(2)        
                    float_int = match.group(3)            
                    float_dec = match.group(4)            
                    float_value = float(f"{float_int}.{float_dec}")  

                    full_path = os.path.join(root, file)
                    matched_files.append(
                        (full_path, system_name, config_string, float_value)
                    )
                else:
                    print(f"Skipped (does not match expected format): {file}")

    matched_files.sort(key=lambda x: x[3])  
    return matched_files

def find_rhf_files_with_metadata_for_fci(folder_path):
    """
    Locate FCI-based RHF files and extract metadata from their filenames.

    The directory tree rooted at ``folder_path`` is searched recursively.
    Files are expected to follow the naming convention

    ``fcidump_fci_<system>_..._<integer>_<decimal>.rhf``.

    The final two integer fields are interpreted as the integer and
    fractional parts of a floating-point parameter such as the bond
    length.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Root directory to search recursively.

    Returns
    -------
    list of tuple
        Tuples of the form ``(file_path, system_name, value)``, sorted
        in ascending order of ``value``.

    Notes
    -----
    Files that do not match the expected naming convention are ignored.
    If ``folder_path`` does not exist, an empty list is returned.
    """
    matched_files = []

    if not os.path.isdir(folder_path):
        print(f"Not found: {folder_path}")
        return []

    pattern = re.compile(
        r'fcidump_fci_([A-Za-z0-9\-]+)_.*_([0-9]+)_([0-9]+)\.rhf$', re.IGNORECASE)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".rhf"):
                match = pattern.search(file)
                if match:
                    system_name = match.group(1)
                    float_int = match.group(2)
                    float_dec = match.group(3)
                    float_value = float(f"{float_int}.{float_dec}")
                    full_path = os.path.join(root, file)
                    matched_files.append((full_path, system_name, float_value))
    matched_files.sort(key=lambda x: x[2])  
    return matched_files

def get_folder_paths_and_values(folder_path):
    """
    Locate numerically named directories and extract their values.

    The directory tree rooted at ``folder_path`` is searched recursively.
    A directory is selected when its name consists entirely of a
    non-negative integer or floating-point number.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Root directory to search recursively.

    Returns
    -------
    sorted_paths : list of str
        Paths to the matched directories, sorted by their numeric values.
    sorted_values : list of float
        Numeric values extracted from the corresponding directory names.

    Notes
    -----
    Directories whose names do not represent numeric values are skipped.
    This function is used, for example, to locate directories associated
    with different random seeds.
    """

    paths = []
    values = []

    # Match folder names that represent numeric values.
    pattern = re.compile(r'^([0-9]+(?:\.[0-9]+)?)$')

    for root, dirs, files in os.walk(folder_path):

        for d in dirs:

            match = pattern.match(d)

            if match:

                float_value = float(match.group(1))

                full_path = os.path.join(root, d)

                paths.append(full_path)
                values.append(float_value)

            else:
                print(f"Skipped: {d}")

    # Sort by the extracted numeric value.
    sorted_pairs = sorted(zip(paths, values), key=lambda x: x[1])

    if sorted_pairs:
        sorted_paths, sorted_values = zip(*sorted_pairs)
    else:
        sorted_paths, sorted_values = [], []

    return list(sorted_paths), list(sorted_values)

def get_statenpz_paths_and_values(folder_path):
    """
    Locate NPZ files and extract numerical values from their filenames.

    The directory tree rooted at ``folder_path`` is searched recursively.
    The function extracts the numerical field immediately preceding the
    ``.npz`` extension.

    For example, ``result_1.50.npz`` is associated with the value ``1.50``.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Root directory to search recursively.

    Returns
    -------
    sorted_paths : list of str
        Paths to the matched NPZ files, sorted by their extracted values.
    sorted_values : list of float
        Numeric values extracted from the corresponding filenames.

    Notes
    -----
    NPZ files that do not match the expected naming convention are
    skipped.
    """
    paths = []
    values = []

    pattern = re.compile(r'_([0-9]+(?:\.[0-9]+)?)\.npz$', re.IGNORECASE)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.npz'):
                match = pattern.search(file)
                if match:
                    float_value = float(match.group(1))
                    full_path = os.path.join(root, file)
                    paths.append(full_path)
                    values.append(float_value)
                else:
                    print(f"Skipped (does not match expected format): {file}")                                                                                                           
    sorted_pairs = sorted(zip(paths, values), key=lambda x: x[1])
    sorted_paths, sorted_values = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return list(sorted_paths), list(sorted_values)

def get_txt_paths_and_values(folder_path):
    """
    Locate TXT files and extract numerical values from their filenames.

    The directory tree rooted at ``folder_path`` is searched recursively.
    The function extracts the numerical field immediately preceding the
    ``.txt`` extension.

    For example, ``output_1.50_0.txt`` is associated with the value ``0``.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Root directory to search recursively.

    Returns
    -------
    sorted_paths : list of str
        Paths to the matched TXT files, sorted by their extracted values.
    sorted_values : list of float
        Numeric values extracted from the corresponding filenames.

    Notes
    -----
    TXT files that do not match the expected naming convention are
    skipped.
    """
    paths = []
    values = []

    pattern = re.compile(r'_([0-9]+(?:\.[0-9]+)?)\.txt', re.IGNORECASE)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                match = pattern.search(file)
                if match:
                    float_value = float(match.group(1))
                    full_path = os.path.join(root, file)
                    paths.append(full_path)
                    values.append(float_value)
                else:
                    print(f"Skipped: does not match expected format: {file}")                                                                                                           
    sorted_pairs = sorted(zip(paths, values), key=lambda x: x[1])
    sorted_paths, sorted_values = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return list(sorted_paths), list(sorted_values)

def load_state_on_main(target_data, anc, main, qu_state):
    """
    Embed a state vector into the main register of a larger quantum state.

    ``target_data`` is interpreted as a state of ``main`` qubits. The
    state is embedded into a register containing ``anc + main`` qubits
    while the ancillary register is initialized to the all-zero state.

    With the qubit ordering used in this repository, the amplitude
    ``target_data[i]`` is placed at index ``2**anc * i`` of the full
    state vector.

    Parameters
    ----------
    target_data : array_like
        State-vector amplitudes for the main register. Its length must be
        ``2**main``.
    anc : int
        Number of ancillary qubits.
    main : int
        Number of main-register qubits.
    qu_state
        Quantum-state object whose state vector is overwritten by the
        embedded state.

    Returns
    -------
    None
        The supplied ``qu_state`` is modified in place.

    Notes
    -----
    The ancillary qubits are assumed to be initialized in the all-zero
    state.
    """
    reserve_for_load = np.zeros(2**(anc+main), dtype=complex)
    for i in range(2**main):
        reserve_for_load[2**anc*i] = target_data[i]
    qu_state.load(reserve_for_load)

def load_target_and_shift(root, index, seed):
    """
    Load the target energy and Hamiltonian shift for one geometry.

    If numerically named seed directories are present under ``root``,
    the directory corresponding to ``seed`` is selected. Otherwise,
    ``root`` itself is treated as a single optimization run.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory containing optimization results.
    index : int
        Index of the molecular geometry in the sorted list of TXT files.
    seed : int or float
        Random seed identifying the optimization run.

    Returns
    -------
    target : float
        Target energy stored under ``"True energy"``.
    shift : float
        Hamiltonian shift stored under ``"Shift value"``.

    Raises
    ------
    FileNotFoundError
        If the requested seed directory or TXT result files cannot be
        found.
    KeyError
        If ``"True energy"`` or ``"Shift value"`` is absent from the
        selected TXT file.
    """
    folders, seeds = seed_folders(root)

    if seeds[0] is not None:
        matched = [
            f for f, s in zip(folders, seeds)
            if np.isclose(s, seed)
        ]

        if not matched:
            raise FileNotFoundError(
                f"No folder for seed {seed} in {root}"
            )

        folders = matched

    txt_files, _ = get_txt_paths_and_values(folders[0])

    if len(txt_files) == 0:
        raise FileNotFoundError(
            f"No txt files found in {folders[0]}"
        )

    txt = txt_files[index]

    return (
        read_value(txt, "True energy"),
        read_value(txt, "Shift value"),
    )

def load_trajectories(root, index):
    """
    Load optimization trajectories for one molecular geometry.

    Optimization histories are collected from all seed directories under
    ``root``. If the trajectories contain different numbers of iterations,
    shorter trajectories are padded with ``NaN`` values.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory containing optimization results.
    index : int
        Index of the molecular geometry in the sorted list of NPZ files.

    Returns
    -------
    trajectories : numpy.ndarray
        Array with shape ``(n_seeds, max_iterations)`` containing the
        optimization trajectories. Missing trailing iterations are
        represented by ``NaN``.
    bond_length : float
        Numerical value associated with the selected geometry.

    Raises
    ------
    FileNotFoundError
        If no NPZ result files are found.
    ValueError
        If the selected geometry is inconsistent across optimization runs.
    """
    histories, bond_lengths = [], []

    for folder in seed_folders(root)[0]:
        files, values = get_statenpz_paths_and_values(folder)

        if len(files) == 0:
            raise FileNotFoundError(f"No npz files found in {folder}")

        histories.append(
            np.asarray(
                np.load(str(files[index]))["array1"],
                dtype=float,
            ).ravel()
        )
        bond_lengths.append(values[index])

    if not np.allclose(bond_lengths, bond_lengths[0]):
        raise ValueError(
            f"Different bond lengths were selected in {root}: "
            f"{set(bond_lengths)}"
        )

    arr = np.full(
        (len(histories), max(len(h) for h in histories)),
        np.nan,
    )

    for j, h in enumerate(histories):
        arr[j, :len(h)] = h

    return arr, bond_lengths[0]

def qubit_hamiltonian(fcidump_path):
    """
    Construct a qubit Hamiltonian from an FCIDUMP-derived RHF file.

    The molecular Hamiltonian is transformed using the Jordan-Wigner
    transformation and decomposed into Pauli operators.

    Parameters
    ----------
    fcidump_path : str or pathlib.Path
        Path to the RHF file used to construct the molecular Hamiltonian.

    Returns
    -------
    coeffs : numpy.ndarray
        Real-valued coefficients of the Pauli decomposition. The first
        coefficient corresponds to the identity operator.
    ops
        Pauli operators corresponding to ``coeffs``.
    jw_h
        Jordan-Wigner-transformed qubit Hamiltonian.

    Raises
    ------
    ValueError
        If any Pauli coefficient has a non-negligible imaginary part.

    Notes
    -----
    ``read_general`` may return complex-valued coefficients even when
    their imaginary parts are numerically zero. Such imaginary parts are
    discarded after verification with an absolute tolerance of ``1e-12``.
    """
    from openfermion.transforms import jordan_wigner
    from qchem_util import get_molecular_hamiltonian_from_fcidump
    from pitbe import read_general

    jw_h = jordan_wigner(
        get_molecular_hamiltonian_from_fcidump(str(fcidump_path))
    )
    coeffs, ops = read_general(str(jw_h))

    coeffs = np.asarray(coeffs, dtype=complex)

    if not np.allclose(coeffs.imag, 0.0, atol=1e-12):
        raise ValueError(
            "Hamiltonian contains non-negligible imaginary coefficients."
        )

    return coeffs.real, ops, jw_h

def read_value(txt_file, key):
    """
    Read a numerical value associated with a key from a TXT file.

    The file is expected to contain entries of the form
    ``<key>: <value>``.

    Parameters
    ----------
    txt_file : str or pathlib.Path
        Path to the TXT file.
    key : str
        Key identifying the value to read.

    Returns
    -------
    float
        Numerical value associated with ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not found in the file.
    """
    with open(txt_file, "r") as f:
        for line in f:
            if line.startswith(key + ":"):
                return float(line.split(":")[1].strip())

    raise KeyError(f"'{key}' not found in {txt_file}")

def seed_folders(root):
    """
    Locate directories corresponding to repeated optimization runs.

    Numerically named subdirectories under ``root`` are interpreted as
    random-seed directories. If no such directories are found, ``root``
    itself is treated as a single optimization run.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory containing optimization results.

    Returns
    -------
    folders : list
        Directories containing individual optimization runs.
    seeds : list
        Numeric seed values corresponding to ``folders``. If no seed
        directories are present, ``[None]`` is returned.
    """
    folders, seeds = get_folder_paths_and_values(root)

    if len(folders) == 0:
        return [root], [None]

    return folders, seeds
