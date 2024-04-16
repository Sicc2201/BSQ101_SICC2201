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
import matplotlib.pyplot as plt

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

    ####  uncomment if this is the first time you connect to your IBMQ account  ##########

    # IBMQ_credentials.ibmq_connexion(ibmq_token)

    #################################################################################

    #_, backend = IBMQ_credentials.ibmq_provider(backend_name)

    backend = IBMQ_credentials.get_local_simulator()
    execute_opts = {'shots': 512}

    num_trotter_steps = np.arange(50)
    time_values = np.arange(0, 5, 0.1)
    theta = np.pi/5


    single_spin_initial_state = qevo.create_single_spin_initial_state()
    single_spin_hamiltonian = qevo.create_single_spin_hamiltonian(theta)
    two_spin_initial_state = qevo.create_two_spin_initial_state()
    two_spin_hamiltonian = qevo.create_two_spin_hamiltonian(theta)

    initial_state = single_spin_initial_state    
    hamiltonian = single_spin_hamiltonian

    observables = qevo.create_observables(initial_state.num_qubits)

    print("computing exact evolution")
    exact_evolution_expected_values = qevo.exact_evolution(initial_state, hamiltonian, time_values, observables)
    print("computing trotterisation")
    trotter_evolution_expected_values = qevo.trotter_evolution(initial_state, hamiltonian, time_values, observables, num_trotter_steps)


    end_time = time.time()
    print('Runtime: ', end_time-start_time, 'sec')


    error = Utils.validate_results(trotter_evolution_expected_values, exact_evolution_expected_values)
    print('MSE: ', error)


    Utils.save_quantum_evolution_plot(time_values, exact_evolution_expected_values.transpose(), trotter_evolution_expected_values.transpose(), observables)
    return 0


if __name__ == "__main__":
    main()