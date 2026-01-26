"""
This module provides utility functions for locating and processing simulation input/output files
used in quantum chemistry and quantum computing experiments.

Functions included:
- find_rhf_files_with_metadata: Extracts metadata from RHF input files with CASCI configuration.
- find_rhf_files_with_metadata_for_fci: Targets RHF files specifically derived from FCI calculations.
- get_statenpz_paths_and_values: Loads `.npz` state files indexed by floating-point geometry values.
- load_state_on_main: Embeds a reduced quantum state into a larger Hilbert space for simulation.

Use cases:
- Batch preparation and organization of input `.rhf` files used in quantum simulations.
- Loading precomputed quantum states for main-register qubits with ancillary padding.
- Pre-sorting datasets by geometric parameters for systematic benchmarking.

Requirements:
- Python 3.7+
- numpy
- os, re (standard library)

Note:
- This script assumes a specific file naming convention for `.rhf` and `.npz` files.
- File paths are expected to be recursively searched from a provided root directory.
- These utilities are typically used as part of a larger simulation workflow.
"""

import numpy as np
import os
import re

def find_rhf_files_with_metadata(folder_path):
    """
    Recursively searches the specified folder for `.rhf` files that match a specific naming pattern,
    extracts metadata (system name, configuration string, and floating-point variable), 
    and returns a sorted list of tuples containing:
      - Full file path
      - System name (e.g., "N2")
      - Number of active spaces and electrons (e.g., "6_6")
      - Floating-point value (e.g., 1.0 from "1_0")

    Expected filename pattern: 
        casci_<System>_..._<Config>_<IntPart>_<DecimalPart>.rhf
    Example:
        casci_N2_active_6_6_1_0.rhf  → ("N2", "6_6", 1.0)

    Parameters:
    ----------
    folder_path : str
        Root directory to start the recursive search.

    Returns:
    -------
    matched_files : list of tuples
        List of (filepath, system_name, config_string, float_value) sorted by float_value.

    Notes:
    -----
    - Skips files not matching the expected pattern.
    - Prints warnings if the folder does not exist or if any file is skipped due to mismatch.
    - Intended for locating RHF files with embedded metadata for simulation input.

    Usage Example:
    --------------
    files = find_rhf_files_with_metadata("data/rhf/")
    for f in files:
        print(f)
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
                    print(f"スキップ（形式不一致）: {file}")

    matched_files.sort(key=lambda x: x[3])  
    return matched_files

def find_rhf_files_with_metadata_for_fci(folder_path):
    """
    Recursively searches the specified folder for `.rhf` files generated from
    FCI calculations and extracts metadata embedded in the filename.

    This function is intended for locating RHF files corresponding to
    FCI-based reference calculations, where the molecular system name and
    a floating-point variable (e.g., bond length) are encoded in the filename.

    Expected filename pattern:
        fcidump_fci_<System>_..._<IntPart>_<DecimalPart>.rhf

    Example:
        fcidump_fci_H2_ccpvdz_1_0.rhf
        → system_name = "H2", float_value = 1.0

    Parameters:
    ----------
    folder_path : str
        Root directory from which the recursive search for `.rhf` files starts.

    Returns:
    -------
    matched_files : list of tuples
        A list of tuples of the form:
            (full_path, system_name, float_value)

        where:
        - full_path (str): absolute path to the `.rhf` file
        - system_name (str): molecular or system identifier extracted from filename
        - float_value (float): numerical parameter reconstructed from filename

        The list is sorted in ascending order of `float_value`.

    Notes:
    -----
    - If the specified folder does not exist, a warning message is printed and
      an empty list is returned.
    - Only files with the `.rhf` extension and matching the expected naming
      pattern are included.
    - This function is primarily used to organize and load FCI reference data
      for comparison with variational or block-encoding-based simulations.

    Usage Example:
    --------------
    files = find_rhf_files_with_metadata_for_fci("data/rhf/fci/")
    for path, system, value in files:
        print(system, value)
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

def get_statenpz_paths_and_values(folder_path):
    """
    Recursively searches for `.npz` files in the specified folder and extracts numeric values from filenames.

    The function looks for filenames that end with a number (including decimal) immediately before the `.npz` extension.
    For example: `npz_N2_1.50.npz` → float value = 1.50

    Returns:
        sorted_paths (List[str]): List of full paths to matched `.npz` files, sorted by the numeric value.
        sorted_values (List[float]): List of extracted numeric values corresponding to the filenames.

    Args:
        folder_path (str): Root folder to search for `.npz` files.

    Notes:
        - Files that do not match the expected format will be skipped with a printed warning.
        - Useful for loading data indexed by geometry or other float-valued parameters.
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
                    print(f"Skipped: does not match expected format: {file}")                                                                                                           
    sorted_pairs = sorted(zip(paths, values), key=lambda x: x[1])
    sorted_paths, sorted_values = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return list(sorted_paths), list(sorted_values)

def load_state_on_main(target_data, anc, main, qu_state):
    """
    Load a quantum state onto the main register of a quantum system.

    This function prepares a quantum state vector for loading into a quantum simulator,
    assuming a system composed of ancillary (anc) and main (mai) qubits. The input
    state `target_data` is assumed to correspond to the amplitudes of the main qubits only,
    and this function embeds it into the full Hilbert space by placing the amplitudes
    in the subspace where all ancilla qubits are initialized to |0⟩.

    Parameters:
        target_data (np.ndarray): A 1D complex-valued numpy array of length 2**mai
            representing the quantum state to be loaded onto the main register.
        anc (int): Number of ancilla qubits. The final state will be embedded into
            a register of size (anc + mai), where ancilla qubits are set to |0⟩.
        main (int): Number of main qubits. Must satisfy len(target_data) == 2**main.
        qu_state (QuantumState): A quantum state object (e.g., from Qulacs) that will
            be overwritten with the new state vector. Its dimension must be 2**(anc + mai).

    Behavior:
        - Creates a zero-initialized state vector of length 2**(anc + mai)
        - Places each amplitude of `target_data[i]` into the basis state where
          ancilla qubits are all zero: position 2**anc * i
        - Loads the constructed state into `qu_state`

    Example:
        If anc = 2 and mai = 1, this will place target_data[0] at index 0,
        and target_data[1] at index 4 (binary: 100), corresponding to:
            |00⟩|0⟩ and |00⟩|1⟩ → embedded in |000⟩ and |100⟩

    Note:
        This assumes the ancilla qubits are in the zero state and not entangled
        with the main qubits initially.
    """
    reserve_for_load = np.zeros(2**(anc+main), dtype=complex)
    for i in range(2**main):
        reserve_for_load[2**anc*i] = target_data[i]
    qu_state.load(reserve_for_load)
