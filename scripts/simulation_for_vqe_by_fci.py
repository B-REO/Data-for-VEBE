"""
This script performs VQE processing and estimates energy values
for Full-CI Hamiltonians transformed via Jordan-Wigner mapping.

This script was used to generate the raw data 
for Fig.3 presented in our manuscript.
The default parameters and simulation settings 
in this script correspond to the conditions described 
at the beginning of Section 3 and in Section 3.1 of the paper.

Output:
- Compressed `.npz` files containing:
    * array1: estimated energy list (per set)
    * array2: variable value

  These are saved to:
    data/npz/worked/vebe/fci/{molecule}/{timestamp}/npz_{molecule}_{distance}.npz

- Log `.txt` files recording for each variable value:
    * RHF input file name
    * number of set
    * analytical expectation value
    * number of Pauli terms
    * number of measurements
    * number of measurements for each Pauli term

  These are saved to:
    data/txt/worked/vebe/fci/{molecule}/{timestamp}/output_{variable}_{index}.txt

Note:
During execution, the program may occasionally terminate due to a temporary
failure in reading the `.rhf` file.
This issue is not caused by corruption of the `.rhf` file itself, but is
attributed to environment-dependent I/O behavior.

If this occurs, simply rerun the program; no modification of the input files is required.
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
    find_rhf_files_with_metadata_for_fci,
    get_statenpz_paths_and_values,
)
from module.sampling import expect_sampling
from module.vqe_opt_alloc import optimal_shots_for_vqe_terms
from qchem_util import get_molecular_hamiltonian_from_fcidump

# === Step 0: Generate timestamp for current execution ===
now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M")

# === Step 1: Locate RHF and NPZ files ===
# Please change "H2" when you simulate other molecule.
rhf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "rhf", "H2")
state_dir = os.path.join(os.path.dirname(__file__), "..", "data", "npz", "for_paper", "state", "H2")
rhf_list = find_rhf_files_with_metadata_for_fci(os.path.expanduser(rhf_dir))
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

    # === Step 2.4: Perform eq.(23) ~ (27) in the paper ===
    re_num, re_ope = pitbe.read_general(str(jw_hamiltonian))
    corrector = 0
    re_num_arr = np.asarray(re_num, dtype=np.complex128).ravel()

    # transform from complex to float
    if np.max(np.abs(re_num_arr.imag)) > 1e-12:
        raise ValueError(f"re_num has non-negligible imaginary parts: max |Im|={np.max(np.abs(re_num_arr.imag))}")
    re_num_arr = re_num_arr.real.astype(float)

    # eq.(23)
    if (eig_min + eig_max - 2*re_num_arr[0]) > 0:
        corrector = (eig_min + eig_max)/2
    re_num_arr[0] -= corrector

    # eq.(24)
    s_dush = np.sum(np.abs(re_num_arr))

    # eq.(25)
    total_shots = np.ceil(s_dush**2/1.6/1.6*10**6/4)
    
    # eq.(27)
    delta = total_shots**0.5*(1.6)/10**5/(s_dush - np.abs(eig_min/2 - eig_max/2))*np.sqrt(s_dush**2 - np.abs(eig_min/2 - eig_max/2)**2)
    
    re_num_arr[0] -= np.abs(delta)
    # eq.(26)
    corrector += np.abs(delta)
    
    # === Step 2.5: Prepare quantum state and perform eq.(28) ===
    # prepare quantum state
    main = pitbe.total_search(re_ope)
    state_vqe = QuantumState(main)
    state_vqe.set_zero_state()
    state_vqe.load(jw_eig_vec)

    # eq.(29)
    shots_per_term, term_vars, term_w = optimal_shots_for_vqe_terms(
        re_num_arr, re_ope, 
        state_vqe, main, 
        total_shots=total_shots, 
        min_shots_nonzero=0, round_mode='int', enforce_total=True
    )

    # === Step 2.6: Measure quantum state and calculate eq.(5) and eq.(6) for all unitary operators===
    loop_time = 194
    vqe_result_list = []
    for j in range(loop_time):
        sub_list = []
        for k in range(len(re_ope)):
            sub_list.append(expect_sampling(re_ope[k], 
                                            int(shots_per_term[k]), 
                                            main, state_vqe))
        vqe_result_list.append(sub_list)

    # === Step 2.7: Save results as NPZ file and TXT file ===
    # NPZ files
    result_dir = os.path.join(os.path.dirname(__file__), 
                              "..", "data", 
                              "npz", "worked", 
                              "vqe", "fci", 
                              str(rhf_list[i][1]), time)
    os.makedirs(result_dir, exist_ok=True)  
    f_name = "npz_" + str(rhf_list[i][1]) + "_" + str(rhf_list[i][3]) + ".npz" 
    file_n = os.path.join(result_dir, f_name)
    np.savez_compressed(file_n, array1=vqe_result_list, array2=rhf_list[i][3])

    # TXT files
    txt_dir = os.path.join(os.path.dirname(__file__), 
                           "..", "data", 
                           "txt", "worked", 
                           "vqe", "fci", 
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
        file.write("Total measurements (VQE): " + str(total_shots) + "\n")
        file.write("Measurements per Pauli term (VQE, all): " + str(list(map(int, shots_per_term))) + "\n")

    print(f"{txt_file} has been created.")

print("Program completed.")