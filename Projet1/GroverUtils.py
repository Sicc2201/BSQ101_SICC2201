##########################################################################

# Titre: GroverUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 09/01/2024

##########################################################################
'''
# Description:

This file contains all the methods that we need to run a full Grover algorithm given logical formulas.


# Methods:

cnf_to_oracle(logical_formula: And) -> Gate : Method that translates a normal conjunctive logical formula into an oracle that takes the form of a quantum gate.

build_diffuser(num_of_vars: int) -> Gate : Method that build the diffuser depending on the number of input qubits.

build_grover_circuit(oracle: Gate, num_of_vars: int, num_iters: int) -> Gate : Method that build the Grover algorithm from the inputs.

solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend: Backend) ->  List[dict]: Given a logical formula, a method to convert
 this formula into an oracle, and a backend on which to execute a quantum circuit.

'''

###########################################################################

# IMPORTS

###########################################################################
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import XGate, ZGate, MCMT, MCMTVChain, Diagonal
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map
from sympy import symbols, Implies, Not, And, Or, to_cnf
from math import floor, sqrt, pi

from typing import Callable


###########################################################################

# Methods

###########################################################################

def initialize_s(qc: QuantumCircuit, first: int, last: int):
    """Apply a H-gate to 'qubits' in qc"""
    for q in range(last - first):
        qc.h(q + first)
    return qc


def reset_variables_circuit():
    return 0


def mesure_qubits(qc, nqubits):
    for i in range(nqubits):
        qc.measure(i, i)

def args_to_toffoli(qc, variables,  clause, index):
    # registers = qc.qregs
    # vreg = registers[0]
    # areg = registers[1]
    print("variables: ", variables)
    toffoli_qubits = ""
    qubit_index = []

    if isinstance(clause, And):
        for i in clause.args:
            if isinstance(i, Not):
                toffoli_qubits += "0"
                qubit_index.append(variables.index(Not(i)))
            else:
                toffoli_qubits += "1"
                qubit_index.append(variables.index(i))
        
    if isinstance(clause, Or):
        for i in clause.args:
            if isinstance(i, Not):
                toffoli_qubits += "1"
                qubit_index.append(variables.index(Not(i)))
            else:
                toffoli_qubits += "0"
                qubit_index.append(variables.index(i))

    else:
        raise ValueError("The clause is not a valid one, it should be of type And or Or")
    
    qubit_index.append(index)
    print("toffoli_qubits: ", toffoli_qubits)
    print("qubit index: ", qubit_index)
    toffoli_gate = XGate().control(len(toffoli_qubits), ctrl_state = toffoli_qubits)
    qc.append(toffoli_gate, qubit_index)

def cnf_to_oracle(logical_formula: And):

    print(logical_formula)
    variables = logical_formula.atoms()
    print("proposition values: ", variables, " of type: ", type(variables),  " of lenght: ", len(variables))

    variables_circuit = QuantumRegister(len(variables), "var_qubits")
    clauses_circuit = QuantumRegister(len(variables), "anc_qubits")

    qc = QuantumCircuit(variables_circuit, clauses_circuit)

    i = len(variables)
    for clause in logical_formula.args:
        print("clause", clause)
        args_to_toffoli(qc, list(variables),  clause, i)
        if isinstance(clause, Or):
            qc.x(i)
        print("*********************  gate applied to the circuit ************************")
        i += 1

    qc.draw("mpl")
    oracle_gate = qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate


def build_grover_circuit(gate, num_of_vars: int, num_iters: int):
    
    variables_circuit = QuantumRegister(num_of_vars, name = "variables")
    clauses_circuit = QuantumRegister(num_of_vars, name = "clauses")
    cr = ClassicalRegister(num_of_vars, name = "CR")

    qc = QuantumCircuit(variables_circuit, clauses_circuit, cr)

    grover_circuit = initialize_s(qc, 0, variables_circuit.size - 1)

    for i in range(num_iters):
        grover_circuit.append(gate, qc.qubits)
        grover_circuit.barrier()
        grover_circuit.append(MCMT("z", clauses_circuit.size - 1, 1), list(range(variables_circuit.size, 2 * clauses_circuit.size))) # apply multicontrolled z gate
        grover_circuit.barrier()
        # *************************
        # Apply the reverse gate
        # **************************
        grover_circuit.append(gate, qc.qubits)
        grover_circuit.append(build_diffuser(variables_circuit.size), list(range(variables_circuit.size)))
        grover_circuit.barrier()

    return grover_circuit


    # *************************
    # TO CHECK
    # **************************

def build_diffuser(num_of_vars: int):
    qc = QuantumCircuit(num_of_vars)

    # apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(num_of_vars):
        qc.h(qubit)

    # apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(num_of_vars):
        qc.x(qubit)

    # simulate a multicontrolled z gate
    qc.h(num_of_vars-1)
    qc.mct(list(range(num_of_vars-1)), num_of_vars-1)  # did not find a way to do a multicontrolled z gate with a gate instruction (MCMT() is not a gate instruction)
    qc.h(num_of_vars-1)

    # apply transformation |11..1> -> |00..0>
    for qubit in range(num_of_vars):
        qc.x(qubit)
        
    # apply transformation |00..0> -> |s>
    for qubit in range(num_of_vars):
        qc.h(qubit)

    U_s = qc.to_gate()
    U_s.name = "Diffuser"
    return U_s


def solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend):

    nb_solution = 1
    nb_iter = floor(pi/4 * sqrt(len(logical_formula.atoms())/nb_solution)) # if you know the number of solutions

    oracle = logical_formula_to_oracle
    grover_circuit = build_grover_circuit(oracle, len(logical_formula.atoms()), nb_iter)

    # measurement
    print("num_qubits = ", len(logical_formula.atoms()))
    mesure_qubits(grover_circuit, len(logical_formula.atoms()))
 

    grover_circuit.draw("mpl")

    result = {0:0}
    return result
