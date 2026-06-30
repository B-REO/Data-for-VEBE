"""
This script optimizes the energy of molecule  
for Full-CI Hamiltonians transformed via the VEBE method 
through arbitrary seed number.

This script was used to generate the raw data 
for Fig.7 presented in our manuscript.
The default parameters and simulation settings 
in this script correspond to the conditions described 
at the beginning of Appendix B of the paper.

Output:
- Compressed `.npz` files containing:
    * array1: trajectory of optimization
    * array2: result of optimization
    * array3: initial state
    * array4: target value of optimization

  These are saved to:
    data/npz/worked/vebe/optimization/{molecule}/{timestamp}/{seed number}/npz_{molecule}_{distance}.npz

- Log `.txt` files recording for each variable value:
    * RHF input file name
    * iteration for optimization
    * result of optimization
    * target value of optimization
    * normalization factor in BE
    * fci energy
    * molecular name
    * seed number
    * variable value

  These are saved to:
    data/txt/worked/vebe/optimization/{molecule}/{timestamp}/{seed number}/output_{variable}_{index}.txt
"""

import math
import os
import sys
import numpy as np
from datetime import datetime
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from pathlib import Path
from qulacs.gate import DenseMatrix
from qulacs.observable import create_observable_from_openfermion_text
from scipy.optimize import minimize
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
from module.loading_tools import find_rhf_files_with_metadata_for_fci
from module.cost_function import cost_vebe
from qchem_util import get_molecular_hamiltonian_from_fcidump

# === Step 0: Generate timestamp for current execution ===
now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M")

# === Step 1: Locate RHF and NPZ files ===
# Please change "H2" when you simulate other molecule.
rhf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "rhf", "H2")
rhf_list = find_rhf_files_with_metadata_for_fci(os.path.expanduser(rhf_dir))
# Random seeds used in the simulations.
# Modify this range if you want to run a different seed set.
seed_list = np.linspace(0, 99, 100)

# === Step 2: Optimization by VEBE ===
for seed_n in seed_list:
    np.random.seed(int(seed_n))
    for i in range(len(rhf_list)):
        print("Elements No." + str(i) + " starts.")
        # === Step 2.1: Get information about Hamiltonian and transform by Jordan-Wigner transformation ===
        fermion_hamiltonian = get_molecular_hamiltonian_from_fcidump(rhf_list[i][0])
        jw_hamiltonian = jordan_wigner(fermion_hamiltonian)
        qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))
        result_list = []

        # === Step 2.2: Calculate minimum and maximum eigenvalue ===
        ham = get_sparse_operator(jw_hamiltonian).tocsr()
        eval_min = eigsh(ham, k=1, which='SA', return_eigenvectors=False)[0]
        eig_min = float(eval_min)
        eval_max = eigsh(ham, k=1, which='LA', return_eigenvectors=False)[0]
        eig_max = float(eval_max)

        # === Step 2.3: Calculate number of qubits and electrons and prepare the initial variable for ansatz ===
        if str(jw_hamiltonian).startswith("("):
            re_num, re_ope = pitbe.read_jw(str(jw_hamiltonian))    
        else:
            re_num, re_ope = pitbe.read_general(str(jw_hamiltonian))

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

        # === Step 2.4: Construct quantum state and gates B, C as defined in Fig. 1 of the paper ===
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

        # calculate the number of qubits and electron
        n_main = pitbe.total_search(re_ope) 
        n_ancilla = int(np.log2(len(alphas)))
        n_electron = int(n_main/2)
        cont_list = []

        # create quantum gates 'B' and 'C' in Fig. 1
        gate = DenseMatrix([j for j in range(n_ancilla)], mat_res)
        opp_gate = DenseMatrix([j for j in range(n_ancilla)], np.diag(opposite_list))

        for j in range(len(re_ope)):
            cont_list.append(pitbe.cont_order(j, n_ancilla))

        # prepare the initial variable
        ant_roop = math.comb(n_main, n_electron)
        in_the_lst = np.random.rand(2*ant_roop)*1e-1

        def callback(x):
            value = cost_vebe(
                        x, n_main, 
                        n_ancilla, 
                        n_electron, 
                        jw_norm, gate,
                        opp_gate, cont_list,
                        re_ope)
            result_list.append(value)
    
        # === Step 2.5: Optimize the variable by the expectation value ===
        res = minimize(
            cost_vebe,
            in_the_lst,
            args=(
                n_main, n_ancilla, 
                n_electron, jw_norm,
                gate, opp_gate, 
                cont_list, re_ope),
            method='BFGS',
            callback=callback
        )

        # === Step 2.6: Save results as NPZ file and TXT file ===
        # NPZ files
        result_dir = os.path.join(os.path.dirname(__file__), 
                                  "..", "data", 
                                  "npz", "worked", 
                                  "vebe", "optimization", 
                                  str(rhf_list[i][1]), time,
                                  str(seed_n))
        os.makedirs(result_dir, exist_ok=True)
        f_name = "npz_" + str(rhf_list[i][1]) + "_" + str(rhf_list[i][2]) + ".npz" 
        file_n = os.path.join(result_dir, f_name)
        np.savez_compressed(file_n, array1=result_list, 
                            array2=res.x, array3=in_the_lst,
                            array4=eig_min)

        # TXT files
        txt_dir = os.path.join(os.path.dirname(__file__), 
                               "..", "data", 
                               "txt", "worked",
                               "vebe", "optimization", 
                               str(rhf_list[i][1]), 
                               time, str(seed_n))
        os.makedirs(txt_dir, exist_ok=True)
        txt_file = os.path.expanduser(txt_dir + "/output_" + str(rhf_list[i][2]) + "_" + str(i)  + ".txt")
        output_directory = os.path.dirname(txt_file)
        os.makedirs(output_directory, exist_ok=True)
        with open(txt_file, "w") as file:
            file.write("Load file: " + str(rhf_dir) + "\n")
            file.write("Optimization times: " + str(len(result_list)) + "\n")
            file.write("Optimization result: " + str(result_list[-1]+corrector) + "\n")
            file.write("Max Absolute Eigenvalue: " + str(max(np.abs(eig_min - corrector), np.abs(eig_max - corrector))) + "\n")
            file.write("Normalization factor: " + str(jw_norm) + "\n")
            file.write("True energy: " + str(eig_min) + "\n")
            file.write("Molecule: " + str(rhf_list[i][1]) + "\n")
            file.write("Seed number: " + str(seed_n) + "\n")
            file.write("Variable Value: " + str(rhf_list[i][2]) + "\n")
            file.write("Shift value: " + str(-corrector) + "\n")
            
        print(f"{txt_file} has been created.")

print("Program completed.")
