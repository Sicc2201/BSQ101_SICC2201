##########################################################################

# Titre: main.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 26/03/2024

##########################################################################
'''
# Description: 

C'est ce fichier qui sera lancé par l'utilisateur pour lancer l'algorithme.

'''

###########################################################################

# IMPORTS

###########################################################################
import os
import time
import numpy as np

# Custom libraries
import IBMQ_credentials
import Quantum_evolution as qevo
import Utils

###########################################################################

# MAIN

###########################################################################


def main():
    start_time = time.time()

    ibmq_token = "put your token here"
    backend_name = "ibmq_qasm_simulator"
    directory_path = "BSQ101_projects/Projet3/project_code/h2_mo_integrals"
    file_paths = [os.path.join(directory_path, file_path) for file_path in os.listdir(directory_path)]

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    #_, backend = IBMQ_credentials.ibmq_provider(backend_name)

    backend = IBMQ_credentials.get_local_simulator()
    execute_opts = {'shots': 512}

    num_qubits = 4
    num_terms = 3
    
    observables = qevo.create_observables(num_qubits)
    num_trotter_steps = 4
    time_values = np.arange(0, 10, 0.1)
    print('time_values: ', time_values)

    hamiltonian = qevo.create_hamiltonian(num_qubits)
    initial_state = qevo.create_initial_state(num_qubits)

    exact_evolution_expected_values = qevo.exact_evolution(initial_state, hamiltonian, time_values, observables)
    trotter_evolution_expected_values = qevo.trotter_evolution(initial_state, hamiltonian, time_values, observables, num_trotter_steps)


    end_time = time.time()
    print('Runtime: ', end_time-start_time, 'sec')


    error = Utils.validate_results(trotter_evolution_expected_values, exact_evolution_expected_values)
    print('MSE: ', error)
    return 0


if __name__ == "__main__":
    main()