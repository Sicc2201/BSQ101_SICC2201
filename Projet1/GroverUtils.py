##########################################################################

# Titre: GroverUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 09/01/2024

##########################################################################
'''
# Description:

This file contains all the methods that we need to run a full Grover algorithm given logical formulas.


# Methods:

args_to_toffoli(qc: QuantumCircuit, variables: list[symbols],  proposition, index: int) : Method that build toffoli gates from propositions to append in the global quantum circuit.

cnf_to_oracle(logical_formula: And) -> Gate : Method that translates a normal conjunctive logical formula into an oracle that takes the form of a quantum gate.

build_diffuser(num_of_vars: int) -> Gate : Method that build the diffuser depending on the number of input qubits.

build_grover_circuit(oracle: Gate, cnf: And, num_iters: int) -> Gate : Method that build the Grover algorithm from the inputs.

solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend: Backend) ->  dict{string:bool}: Given a logical formula, a method to convert
 this formula into an oracle, and a backend on which to execute a quantum circuit.

'''

###########################################################################

# IMPORTS

###########################################################################

import QuantumUtils as utils

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ, Aer, transpile, execute
from qiskit.circuit.library import XGate, ZGate, MCMT, MCMTVChain, Diagonal
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map
from sympy import symbols, Implies, Not, And, Or, to_cnf
from math import floor, sqrt, pi

from typing import Callable



###########################################################################

# Methods

###########################################################################


def args_to_toffoli(qc: QuantumCircuit, variables: list,  proposition, index: int):

    print("variables: ", variables)
    toffoli_qubits = ""
    qubit_index = []
    print("proposition type: ",type( proposition))

    # Pincus
    if isinstance(proposition, And):
        for i in proposition.args:
            if isinstance(i, Not):
                toffoli_qubits += "0"
                qubit_index.append(variables.index(Not(i)))
            else:
                toffoli_qubits += "1"
                qubit_index.append(variables.index(i))
    
    # Cake problem
    elif isinstance(proposition, Or):

        for i in proposition.args:
            if isinstance(i, Not):
                toffoli_qubits += "1"
                qubit_index.append(variables.index(Not(i)))
            else:
                toffoli_qubits += "0"
                qubit_index.append(variables.index(i))

    else:
        raise ValueError("problem")
    
    qubit_index.append(index)
    print("toffoli_qubits: ", toffoli_qubits)
    print("qubit index: ", qubit_index)
    toffoli_gate = XGate().control(len(toffoli_qubits), ctrl_state = toffoli_qubits[::-1])
    qc.append(toffoli_gate, qubit_index)


def cnf_to_oracle(logical_formula: And):

    print(logical_formula)
    variables = sorted(logical_formula.atoms(), key=lambda x: x.name)
    print("proposition values: ", variables, " of type: ", type(variables),  " of lenght: ", len(variables))

    variables_circuit = QuantumRegister(len(variables), "var_qubits")
    clauses_circuit = QuantumRegister(len(logical_formula.args), "anc_qubits")

    qc = QuantumCircuit(variables_circuit, clauses_circuit)

    i = len(variables)
    for clause in logical_formula.args:
        print("clause", clause)
        args_to_toffoli(qc, list(variables),  clause, i)
        if isinstance(clause, Or):
            qc.x(i)
        print("*********************  oracle part " + str(i - len(variables)) + " ************************")
        i += 1

    qc.draw("mpl")
    oracle_gate = qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate


def build_grover_circuit(gate, cnf, num_iters: int):
    
    num_of_vars = len(cnf.atoms())
    variables_circuit = QuantumRegister(num_of_vars, name = "variables")
    clauses_circuit = QuantumRegister(len(cnf.args), name = "clauses")
    cr = ClassicalRegister(num_of_vars, name = "CR")
    qc = QuantumCircuit(variables_circuit, clauses_circuit, cr)

    grover_circuit = utils.initialize_s(qc, 0, variables_circuit.size)

    for i in range(num_iters):
        grover_circuit.append(gate, qc.qubits)
        grover_circuit.barrier()
        grover_circuit.append(MCMT("z", clauses_circuit.size - 1, 1), list(range(variables_circuit.size, 2 * clauses_circuit.size))) # apply multicontrolled z gate
        grover_circuit.barrier()
        grover_circuit.append(gate.inverse(), qc.qubits)
        grover_circuit.append(build_diffuser(variables_circuit.size), list(range(variables_circuit.size)))
        grover_circuit.barrier()

    grover_circuit.draw("mpl")
        
    return grover_circuit


def build_diffuser(num_of_vars: int):
    qc = QuantumCircuit(num_of_vars)

    qc.h(qc.qubits)
    qc.x(qc.qubits)

    # simulate a multicontrolled z gate
    qc.h(num_of_vars-1)
    qc.mct(list(range(num_of_vars-1)), num_of_vars-1)  # did not find a way to do a multicontrolled z gate with a gate instruction (MCMT() is not a gate instruction)
    qc.h(num_of_vars-1)

    qc.x(qc.qubits)
    qc.h(qc.qubits)

    qc.draw("mpl")

    U_s = qc.to_gate()
    U_s.name = "Diffuser"
    return U_s


def solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend):

    nb_solution = 1
    nb_qubits = len(logical_formula.atoms())
    nb_iter = floor(pi/4 * sqrt(nb_qubits/nb_solution))
    nb_iter = 2
    print("num_iterations = ", nb_iter)
    cnf_atoms = sorted(logical_formula.atoms(), key=lambda x: x.name)

    grover_circuit = build_grover_circuit(logical_formula_to_oracle, logical_formula, nb_iter)

    # measurement
    utils.mesure_qubits(grover_circuit, len(logical_formula.atoms()))

    # Simulate and plot results
    transpiled_qc = transpile(grover_circuit, backend)
    job = backend.run(transpiled_qc)

    results = list(job.result().get_counts().items())
    plot_histogram(job.result().get_counts())

    # print(results)

    boolean_solutions = utils.quantum_results_to_boolean(results, cnf_atoms)

    print(boolean_solutions)

    return boolean_solutions


