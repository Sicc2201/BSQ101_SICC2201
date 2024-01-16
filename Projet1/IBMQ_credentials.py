
##########################################################################

# Titre: IBMQ_credentials.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 10/01/2024

##########################################################################
'''
# Description: 
'''

###########################################################################

# IMPORTS

###########################################################################

# importing Qiskit
from qiskit_ibm_runtime import QiskitRuntimeService


###########################################################################

# Utility

###########################################################################


def ibmq_connexion(ID):
    # Save an IBM Quantum account and set it as your default account.
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=ID, set_as_default=True, overwrite=True)


def ibmq_provider(provider_name):

    provider = QiskitRuntimeService()
    provider.backends()  # list of backends
    backend = provider.backend(provider_name)
    # print(backend.configuration().basis_gates)
    return provider, backend