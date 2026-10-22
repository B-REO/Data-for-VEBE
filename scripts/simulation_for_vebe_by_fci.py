"""
This script performs VEBE processing and estimates energy values
for Full-CI Hamiltonians transformed via Jordan-Wigner mapping.

This script was used to generate the raw data 
for Fig. 3 and 9 presented in our manuscript.
The default parameters and simulation settings 
in this script correspond to the conditions described 
at the beginning of Section 3, in Section 3.A, 
and in Appendix C of the paper.

Output:
- Compressed `.npz` files containing:
    * array1: estimated energy list (per set)
    * array2: correction value
    * array3: analytical probability
    * array4: estimated probability list (per set)

  These are saved to:
    data/npz/worked/vebe/fci/{shift_type}/{molecule}/{timestamp}/npz_{molecule}_{distance}.npz

- Log `.txt` files recording for each variable value:
    * RHF input file name
    * number of set
    * normalization factor
    * analytical energy
    * analytical probability
    * molecular name
    * number of Pauli term
    * number of measurements
    * variable value
    * shift type
    * shift value 
    * delta value

  These are saved to:
    data/txt/worked/vebe/fci/{shift_type}/{molecule}/{timestamp}/output_{variable}_{index}.txt
"""

import math
import os
import sys
import numpy as np
from datetime import datetime
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from pathlib import Path
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import DenseMatrix
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
    load_state_on_main,
)
from module.sampling import batched_sampling_est_prob
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

# === Step 2: Perform VEBE ===
for i in range(len(rhf_list)):
    print("Elements No." + str(i) + " starts.")
    # === Step 2.1: Get information about Hamiltonian and transform by Jordan-Wigner transformation ===
    fermion_hamiltonian = get_molecular_hamiltonian_from_fcidump(rhf_list[i][0])
    jw_hamiltonian = jordan_wigner(fermion_hamiltonian)
    
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
        # Please change the value of electron (ex. "H4":4, "H6":6) when you simulate other molecule.
        electron = 2
        hf_state[2**electron-1] = 1
        state_test.load(hf_state)
        qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))
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
    
    # eq.(41)
    if shift_type == "exact":
        delta = total_shots**0.5*(1.6)/10**5/(s_dush - np.abs(eig_min/2 - eig_max/2))*np.sqrt(s_dush**2 - np.abs(eig_min/2 - eig_max/2)**2) 
    else:
        delta = total_shots**0.5*(1.6)/10**5/(s_dush - np.abs(eig_min_2/2 - eig_max_2/2))*np.sqrt(s_dush**2 - np.abs(eig_min_2/2 - eig_max_2/2)**2)
    shifted_coeffs[0] -= np.abs(delta)

    # eq.(40)
    corrector += np.abs(delta)
    
    jw_norm = np.sum(np.abs(shifted_coeffs))

    # === Step 2.6: Construct quantum state and gates B, C as defined in Fig. 1 of the paper ===
    # calculate $\sqrt{\alpha_i/S}$
    alphas = (np.sqrt(np.abs(shifted_coeffs)) / np.sqrt(np.sum(np.abs(shifted_coeffs))))
    if math.floor(np.log2(len(alphas))) != np.log2(len(alphas)):
        zero_list = np.zeros(2**(math.floor(np.log2(len(alphas))) + 1) - len(alphas))
        alphas = np.append(alphas, zero_list)
    opposite_list = np.ones(len(alphas))
    for j in range(len(shifted_coeffs)):
        if (shifted_coeffs[j].real < 0):
            opposite_list[j] = -1
    cf = pitbe.coeff_make(alphas)
    mat_res = pitbe.mat_maker(alphas, cf)

    main = pitbe.total_search(re_ope)
    anci = int(np.log2(len(alphas)))
    n = anci + main
    cont_list = []

    for j in range(len(re_ope)):
        cont_list.append(pitbe.cont_order(j, anci))

    # prepare quantum state
    state = QuantumState(n)
    state.set_zero_state()
    load_state_on_main(jw_eig_vec, anci, main, state)

    # create quantum gates 'B' and 'C' in Fig. 1
    gate = DenseMatrix([j for j in range(anci)], mat_res)
    opp_gate = DenseMatrix([j for j in range(anci)], np.diag(opposite_list))
    gate_dag = gate.get_inverse()

    # === Step 2.7: Prepare the quantum state and perform eq.(11), (12), (14), and (15) in the paper ===
    # prepare the quantum circuit for VEBE 
    circ = QuantumCircuit(n)

    # quantum gate 'B' for eq.(11)
    circ.add_gate(gate)
    circ.add_gate(opp_gate)

    # quantum gates for eq.(12)
    for j in range(len(cont_list)):    
        pitbe.circ_make(re_ope[j], cont_list[j], circ, main + anci, anci)

    # quantum gate 'C' for eq.(14)
    circ.add_gate(gate_dag)

    # apply the quantum circuit to transform the quantum state
    circ.update_quantum_state(state)

    # perform eq.(15)
    obser_order = []
    for j in range(anci):
        obser_order.append(0)
    for j in range(main):
        obser_order.append(2)
    prob = state.get_marginal_probability(obser_order)

    # === Step 2.8: Measure quantum state and calculate eq.(15) and eq.(17) ===
    loop_time = 194
    energy_list = []
    prob_list = []
    
    # measure quantum state and perform eq.(15)
    probs = batched_sampling_est_prob(state, anci, total_shots, loop_time, max_batch_shots=2000000)
    prob_list = probs.tolist()

    # calculate energy estimator (eq.(17))
    energy_list = (-np.sqrt(probs) * jw_norm).tolist()

    # === Step 2.9: Save results as NPZ file and TXT file ===
    # NPZ files
    result_dir = os.path.join(os.path.dirname(__file__), 
                              "..", "data", 
                              "npz", "worked", 
                              "vebe", "fci",
                              shift_type, 
                              str(rhf_list[i][1]), time)
    os.makedirs(result_dir, exist_ok=True)  
    f_name = "npz_" + str(rhf_list[i][1]) + "_" + str(rhf_list[i][2]) + ".npz" 
    file_n = os.path.join(result_dir, f_name)
    np.savez_compressed(file_n, array1=energy_list, 
                        array2=corrector, array3=prob,
                        array4=prob_list)

    # TXT files
    txt_dir = os.path.join(os.path.dirname(__file__), 
                           "..", "data", 
                           "txt", "worked", 
                           "vebe", "fci",
                           shift_type,
                           str(rhf_list[i][1]),time)
    os.makedirs(txt_dir, exist_ok=True)  
    txt_file = os.path.expanduser(txt_dir + "/output_" + str(rhf_list[i][2]) + "_" + str(i)  + ".txt")
    output_directory = os.path.dirname(txt_file)
    os.makedirs(output_directory, exist_ok=True)
    with open(txt_file, "w") as file:
        file.write("Load file: " + str(rhf_dir) + "\n")
        file.write("Loop times: " + str(loop_time) + "\n")
        file.write("Normalization factor: " + str(jw_norm) + "\n")
        file.write("True energy: " + str(-np.sqrt(prob)*jw_norm) + "\n")
        file.write("True probability: " + str(prob) + "\n")
        file.write("Molecule: " + str(rhf_list[i][1]) + "\n")
        file.write("The number of Pauli rotation gates: " + str(len(re_ope)) + "\n")
        file.write("The number of measurements: " + str(total_shots) + "\n")
        file.write("Variable Value: " + str(rhf_list[i][2]) + "\n")
        file.write("Shift type: " + shift_type + "\n")
        file.write("Shift value: " + str(-corrector) + "\n")
        file.write("Delta: " + str(delta) + "\n")

    print(f"{txt_file} has been created.")

print("Program completed.")
