"""
This script performs VQE processing and estimates energy values
for CASCI-active space Hamiltonians transformed via Jordan-Wigner mapping.

This script was used to generate the raw data 
for Fig.5 and 12 presented in our manuscript.
The default parameters and simulation settings 
in this script correspond to the conditions described 
at the beginning of Section 3, in Section 3.B, 
and in Appendix C of the paper.

Output:
- Compressed `.npz` files containing:
    * array1: sampling results for each set and each Pauli term,
              given as (sampled probability, sampled expectation,
              exact probability, exact expectation)
    * array2: variable value (bond length)

  These are saved to:
    data/npz/worked/vqe/casci/{shift_type}/{molecule}/{timestamp}/npz_{molecule}_{distance}.npz

- Log `.txt` files recording for each variable value:
    * RHF input file name
    * number of set
    * analytical expectation value
    * number of Pauli terms
    * shift type
    * shift value 
    * number of measurements
    * number of measurements for each Pauli term

  These are saved to:
    data/txt/worked/vqe/casci/{shift_type}/{molecule}/{timestamp}/output_{variable}_{index}.txt
"""

import os
import sys
import numpy as np
from datetime import datetime
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from pathlib import Path
from qulacs import QuantumState
from qulacs.observable import create_observable_from_openfermion_text
from scipy.sparse.linalg import eigsh

BASE_DIR = Path(__file__).resolve().parent
PROJECT  = BASE_DIR.parent

sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "vendor" / "PItBE"))
sys.path.insert(
    0,
    str(PROJECT / "vendor" / "quantum_software_handson" / "doc" / "source" / "notebooks")
)
import pitbe
from module.loading_tools import (
    find_rhf_files_with_metadata,
    get_statenpz_paths_and_values,
)
from module.sampling import expect_sampling
from module.vqe_opt_alloc import optimal_shots_for_vqe_terms
from qchem_util import get_molecular_hamiltonian_from_fcidump

# === Step 0: Generate timestamp for current execution ===
now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M")

# === Step 1: Locate RHF and NPZ files ===
rhf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "rhf", "N2")
state_dir = os.path.join(os.path.dirname(__file__), "..", "data", "npz", "for_paper", "state", "N2")
rhf_list = find_rhf_files_with_metadata(os.path.expanduser(rhf_dir))
loadnpz = get_statenpz_paths_and_values(state_dir)

# === Step 2: Perform VQE ===
for i in range(len(rhf_list)):
    print("Elements No." + str(i) + " starts.")
    # === Step 2.1: Get information about Hamiltonian and transform by Jordan-Wigner transformation ===
    fermion_hamiltonian = get_molecular_hamiltonian_from_fcidump(rhf_list[i][0])
    jw_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))

    # === Step 2.2: Load quantum state ===
    all_result = np.load('' + str(loadnpz[0][i]))
    jw_eig_vec = all_result['array5']

    # === Step 2.3: Calculate minimum and maximum eigenvalue ===
    ham = get_sparse_operator(jw_hamiltonian).tocsr()
    eval_min, evec_min = eigsh(ham, k=1, which='SA') 
    eig_min = float(eval_min[0])
    eval_max, evec_max = eigsh(ham, k=1, which='LA') 
    eig_max = float(eval_max[0])

    # === Step 2.4: Choose shift type ===
    shift_type = "exact"
    # Choose "exact" or "approximate".

    if shift_type not in {"exact", "approximate"}:
        raise ValueError(
            f"Unknown shift type: {shift_type!r}. "
            "Choose 'exact' or 'approximate'."
        )

    # === Step 2.5: Perform eq.(37) ~ (41) in the paper ===
    re_num, re_ope = pitbe.read_general(str(jw_hamiltonian))
    corrector = 0
    re_num_arr = np.asarray(re_num, dtype=np.complex128).ravel()

    # transform from complex to float
    if np.max(np.abs(re_num_arr.imag)) > 1e-12:
        raise ValueError(f"re_num has non-negligible imaginary parts: max |Im|={np.max(np.abs(re_num_arr.imag))}")
    re_num_arr = re_num_arr.real.astype(float)

    # eq.(37)
    # Determine the shift used to define the VEBE measurement budget.
    if shift_type == "exact":
        corrector = (eig_min + eig_max)/2 
    else:
        # HF energy
        sub_main = pitbe.total_search(re_ope)
        state_test = QuantumState(sub_main)
        state_test.set_zero_state()
        hf_state = np.zeros(2**sub_main)
        electron = 6
        hf_state[2**electron-1] = 1
        state_test.load(hf_state)
        eig_min_2 = qulacs_hamiltonian.get_expectation_value(state_test)
        # Upper bound of the maximum eigenvalue
        eig_max_2 = re_num_arr[0] + np.sum(np.abs(re_num_arr[1:]))
        corrector = (eig_min_2 + eig_max_2)/2
    
    # Shifted coefficient norm used to determine the VEBE measurement budget
    shifted_coeffs = re_num_arr.copy()
    shifted_coeffs[0] -= corrector

    # eq.(38)
    s_dush = np.sum(np.abs(shifted_coeffs))

    # eq.(39)
    total_shots = np.ceil(s_dush**2/1.6/1.6*10**6/4) 

    # === Step 2.6: Prepare quantum state and perform eq.(42) ===
    # prepare quantum state
    main = pitbe.total_search(re_ope)
    state_vqe = QuantumState(main)
    state_vqe.set_zero_state()
    state_vqe.load(jw_eig_vec)

    # eq.(42)
    shots_per_term, term_vars, term_w = optimal_shots_for_vqe_terms(
        re_num_arr, re_ope, 
        state_vqe, main, 
        total_shots=total_shots, 
        min_shots_nonzero=0, round_mode='int', enforce_total=True
    )

    # === Step 2.7: Measure quantum state and calculate eq.(5) and eq.(6) for all unitary operators===
    loop_time = 194
    vqe_result_list= []
    for j in range(loop_time):
        sub_list = []
        for k in range(len(re_ope)):
            sub_list.append(expect_sampling(re_ope[k], 
                                            int(shots_per_term[k]), 
                                            main, state_vqe))
        vqe_result_list.append(sub_list)

    # === Step 2.8: Save results as NPZ file and TXT file ===
    # NPZ files
    result_dir = os.path.join(os.path.dirname(__file__), 
                              "..", "data", 
                              "npz", "worked", 
                              "vqe", "casci",
                              shift_type,
                              str(rhf_list[i][1]), time)
    os.makedirs(result_dir, exist_ok=True)  
    f_name = "npz_" + str(rhf_list[i][1]) + "_" + str(rhf_list[i][3]) + ".npz" 
    file_n = os.path.join(result_dir, f_name)
    np.savez_compressed(file_n, array1=vqe_result_list, array2=rhf_list[i][3])

    # TXT files
    txt_dir = os.path.join(os.path.dirname(__file__), 
                           "..", "data", 
                           "txt", "worked", 
                           "vqe", "casci", 
                           shift_type, 
                           str(rhf_list[i][1]), time)
    os.makedirs(txt_dir, exist_ok=True)   
    txt_file = os.path.expanduser(txt_dir + "/output_" + str(rhf_list[i][2]) + "_" + str(i)  + ".txt")
    output_directory = os.path.dirname(txt_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(txt_file, "w") as file:
        file.write("Load file: " + str(rhf_dir) + "\n")
        file.write("Loop times: " + str(loop_time) + "\n")
        file.write("True expectation: " + str(qulacs_hamiltonian.get_expectation_value(state_vqe)) + "\n")
        file.write("Number of terms in Hamiltonian: " + str(len(re_ope)) + "\n")
        file.write("Shift type: " + shift_type + "\n")
        file.write("Shift value: " + str(-corrector) + "\n")
        file.write("Total measurements (VQE): " + str(total_shots) + "\n")
        file.write("Measurements per Pauli term (VQE, all): " + str(list(map(int, shots_per_term))) + "\n")

    print(f"{txt_file} has been created.")

print("Program completed.")
