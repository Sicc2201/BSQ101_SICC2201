
##########################################################################

# Titre: IBMQ_credentials.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 21/01/2024

##########################################################################
'''
# Description: 

This file contains all the necessary methods for IBMQ connexion and services

# Methods:

- ibmq_connexion(ID: str) : Saves an IBM Quantum account and set it as your default account. To use only the fist time you connect to your account.

- ibmq_provider(provider_name: str) : Gives you access to the intended provider and backend.
'''

###########################################################################

# IMPORTS

###########################################################################

# importing Qiskit
from qiskit_ibm_runtime import QiskitRuntimeService


###########################################################################

# Methods

###########################################################################


def ibmq_connexion(ID: str):
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=ID, set_as_default=True, overwrite=True)


def ibmq_provider(provider_name: str):

    provider = QiskitRuntimeService()
    provider.backends()  # list of backends
    backend = provider.backend(provider_name)
    # print(backend.configuration().basis_gates)
    return provider, backend