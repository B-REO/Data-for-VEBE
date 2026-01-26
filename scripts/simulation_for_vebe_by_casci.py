"""
This script performs VEBE processing and estimates energy values
for CASCI-active space Hamiltonians transformed via Jordan-Wigner mapping.

This script was used to generate the raw data 
for Fig.5 presented in our manuscript.
The default parameters and simulation settings 
in this script correspond to the conditions described 
at the beginning of Section 3 and in Section 3.2 of the paper.

Output:
- Compressed `.npz` files containing:
    * array1: estimated energy list (per set)
    * array2: correction value
    * array3: analytical probability
    * array4: estimated probability list (per set)

  These are saved to:
    data/npz/worked/vebe/casci/{molecule}/{timestamp}/npz_{molecule}_{distance}.npz

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
    * correction value (If it is not 0.)

  These are saved to:
    data/txt/worked/vebe/casci/{molecule}/{timestamp}/output_{variable}_{index}.txt

Note:
During execution, the program may occasionally terminate due to a temporary
failure in reading the `.rhf` file.
This issue is not caused by corruption of the `.rhf` file itself, but is
attributed to environment-dependent I/O behavior.

If this occurs, simply rerun the program; no modification of the input files is required.
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
    load_state_on_main,
)
from module.sampling import batched_sampling_est_prob
from qchem_util import get_molecular_hamiltonian_from_fcidump

# === Step 0: Generate timestamp for current execution ===
now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M")

# === Step 1: Locate RHF and NPZ files ===
rhf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "rhf", "N2")
state_dir = os.path.join(os.path.dirname(__file__), "..", "data", "npz", "for_paper", "state", "N2")
rhf_list = find_rhf_files_with_metadata(os.path.expanduser(rhf_dir))
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

    # === Step 2.4: Perform eq.(23) ~ (27) in the paper ===
    re_num, re_ope = pitbe.read_general(str(jw_hamiltonian))
    corrector = 0

    # eq.(23)
    if (eig_min + eig_max - 2*re_num[0].real) > 0:
        corrector = (eig_min + eig_max)/2 
    re_num[0] -= corrector 

    # eq.(24)
    s_dush = np.sum(np.abs(re_num)) 

    # eq.(25)
    total_shots = np.ceil(s_dush**2/1.6/1.6*10**6/4) 

    # eq.(27)
    delta = total_shots**0.5*(1.6)/10**5/(s_dush - np.abs(eig_min/2 - eig_max/2))*np.sqrt(s_dush**2 - np.abs(eig_min/2 - eig_max/2)**2) 
    
    re_num[0] -= np.abs(delta)
    # eq.(26)
    corrector += np.abs(delta) 

    jw_norm = np.sum(np.abs(re_num))

    # === Step 2.5: Construct quantum state and gates B, C as defined in Fig. 1 of the paper ===
    # calculate $\sqrt{\alpha_i/S}$
    alphas = (np.sqrt(np.abs(re_num)) / np.sqrt(np.sum(np.abs(re_num))))
    if math.floor(np.log2(len(alphas))) != np.log2(len(alphas)):
        zero_list = np.zeros(2**(math.floor(np.log2(len(alphas))) + 1) - len(alphas))
        alphas = np.append(alphas, zero_list)
    opposite_list = np.ones(len(alphas))
    for j in range(len(re_num)):
        if (re_num[j].real < 0):
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

    # === Step 2.6: Prepare the quantum state and perform eq.(10) ~ (12), (14) in the paper ===
    # prepare the quantum circuit for VEBE 
    circ = QuantumCircuit(n)
    
    # quantum gate 'B' for eq.(10)
    circ.add_gate(gate)
    circ.add_gate(opp_gate)

    # quantum gates for eq.(11)
    for j in range(len(cont_list)):    
        pitbe.circ_make(re_ope[j], cont_list[j], circ, main + anci, anci)
    
    # quantum gate 'C' for eq.(12)
    circ.add_gate(gate_dag)

    # apply the quantum circuit to transform the quantum state
    circ.update_quantum_state(state)

    # perform eq.(14)
    obser_order = []
    for j in range(anci):
        obser_order.append(0)
    for j in range(main):
        obser_order.append(2)
    prob = state.get_marginal_probability(obser_order)

    # === Step 2.7: Measure quantum state and calculate eq.(13) and eq.(15) ===
    loop_time = 194
    energy_list = []
    prob_list = []
    
    # measure quantum state and perform eq.(13)
    probs = batched_sampling_est_prob(state, anci, total_shots, loop_time, max_batch_shots=2000000)
    prob_list = probs.tolist()

    # calculate energy estimator (Eq. 15)
    energy_list = (-np.sqrt(probs) * jw_norm).tolist()

    # === Step 2.8: Save results as NPZ file and TXT file ===
    # NPZ files
    result_dir = os.path.join(os.path.dirname(__file__), 
                              "..", "data", 
                              "npz", "worked", 
                              "vebe", "casci", 
                              str(rhf_list[i][1]), time)
    os.makedirs(result_dir, exist_ok=True)  
    f_name = "npz_" + str(rhf_list[i][1]) + "_" + str(rhf_list[i][3]) + ".npz" 
    file_n = os.path.join(result_dir, f_name)
    np.savez_compressed(file_n, array1=energy_list, 
                        array2=corrector, array3=prob,
                        array4=prob_list)

    # TXT files
    txt_dir = os.path.join(os.path.dirname(__file__), 
                           "..", "data", 
                           "txt", "worked", 
                           "vebe", "casci",
                           str(rhf_list[i][1]), time)
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
        if corrector != 0:
            file.write("Shift value: " + str(-corrector) + "\n")

    print(f"{txt_file} has been created.")

print("Program completed.")
