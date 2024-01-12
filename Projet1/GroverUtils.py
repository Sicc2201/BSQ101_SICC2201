##########################################################################

# Titre: GroverUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 09/01/2024

##########################################################################
'''
# Description:

This file contains all the methods that we need to run a full Grover algorithm given logical formulas.


# Methods:

cnf_to_oracle(logical_formula: And) ->  : Method that translates a normal conjunctive logical formula into an oracle that takes the form of a quantum gate.

build_diffuser(num_of_vars: int) ->  : Method that build the diffuser depending on the number of input qubits.

build_grover_circuit(oracle: Gate, num_of_vars: int, num_iters: int) -> Gate : Method that build the Grover algorithm from the inputs.

solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend: Backend) ->  List[dict]: Given a logical formula, a method to convert
 this formula into an oracle, and a backend on which to execute a quantum circuit.

'''

###########################################################################

# IMPORTS

###########################################################################
import qiskit


###########################################################################

# Methods

###########################################################################

def cnf_to_oracle(logical_formula):
    gate = 0
    return gate

def build_diffuser(num_of_vars):
    diffuser = 0
    return diffuser

def build_grover_circuit(gate, num_of_vars, num_iters):
    grover_circuit = 0
    return grover_circuit

def solve_sat_with_grover(logical_formula, logical_formula_to_oracle: cnf_to_oracle(), backend):
    result = {0:0}
    return result