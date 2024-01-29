##########################################################################

# Titre: GroverUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 23/01/2024

##########################################################################
'''
# Description:

This file contains all the methods that we need to run a full Grover algorithm given logical formulas.


# Methods:

- disjonction_gate(variables: list,  proposition: Or, index: int) -> Tuple[CCXGate, str] : create the toffolis gate according to the proposition.

- create_oracle_gates(logical_formula: And) -> Gate : Create the parts of the oracles depending on the toffolis build from disjonction_gate().

- cnf_to_oracle(logical_formula: And) -> Gate : Translates a normal conjunctive logical formula into an oracle that takes the form of a quantum gate.

- build_diffuser(num_of_vars: int) -> Gate : Build the diffuser depending on the number of input qubits.

- build_grover_circuit(oracle: Gate, num_of_vars: int, num_iters: int) -> QuantumCircuit : Build the Grover algorithm from the inputs.

- solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend: Backend) ->  dict{string:bool}: Given a logical formula, converts
    this formula into an oracle, and a backend on which to execute a quantum circuit.

- validate_grover_solutions(results: list[dict], cnf: And) : validate the result from the sumulation with sympy.

'''

###########################################################################

# IMPORTS

###########################################################################

import QuantumUtils as utils

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, ZGate, CCXGate
from sympy import Not, And, Or
from math import floor, sqrt, pi

from typing import Callable, Tuple



###########################################################################

# Methods

###########################################################################


def disjonction_gate(variables: list,  proposition: Or, index: int) -> Tuple[CCXGate, str]:

    toffoli_qubits = ""
    qubit_index = []

    for i in proposition.args:
        if isinstance(i, Not):
            toffoli_qubits += "1"
            qubit_index.append(variables.index(Not(i)))
        else:
            toffoli_qubits += "0"
            qubit_index.append(variables.index(i))

    qubit_index.append(index)
    toffoli_gate = XGate().control(len(toffoli_qubits), ctrl_state = toffoli_qubits[::-1]) # [::-1] inverse the string to respect little endian

    return toffoli_gate, qubit_index


def create_oracle_gates(logical_formula: And) -> Gate:

    # sort the proposition atoms
    variables = sorted(logical_formula.atoms(), key=lambda x: x.name)

    variables_circuit = QuantumRegister(len(variables), "var_qubits")
    clauses_circuit = QuantumRegister(len(logical_formula.args), "anc_qubits")
    qc = QuantumCircuit(variables_circuit, clauses_circuit)

    # Apply the right toffoli gate to the circuit
    i = len(variables)
    for clause in logical_formula.args:

        toffoli_gate, qubit_index = disjonction_gate(list(variables),  clause, i) 
        qc.append(toffoli_gate, qubit_index)
        qc.x(i)

        i += 1

    # create the oracle gate part
    oracle_gate = qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate


def cnf_to_oracle(logical_formula: And) -> Gate:

    num_vars = len(logical_formula.atoms())
    num_clauses = len(logical_formula.args)

    variables_circuit = QuantumRegister(num_vars, name = "variables")
    clauses_circuit = QuantumRegister(num_clauses, name = "clauses")
    qc = QuantumCircuit(variables_circuit, clauses_circuit)

    gate = create_oracle_gates(logical_formula)
    z_gate = ZGate().control(num_clauses - 1)
    

    qc.append(gate, qc.qubits)
    qc.append(z_gate, range(num_vars, qc.num_qubits))
    qc.append(gate.inverse(), qc.qubits)
    

    # create the oracle gate
    oracle_gate = qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate


def build_grover_circuit(oracle: Gate, num_of_vars: int, num_iters: int) -> QuantumCircuit:

    # create the global circuit
    variables_circuit = QuantumRegister(num_of_vars, name = "variables")
    clauses_circuit = QuantumRegister(oracle.num_qubits - num_of_vars, name = "clauses")
    cr = ClassicalRegister(num_of_vars, name = "CR")
    grover_circuit = QuantumCircuit(variables_circuit, clauses_circuit, cr)

    # initialize circuit |s> state 
    grover_circuit.h(range(variables_circuit.size))
    diffuser = build_diffuser(num_of_vars)

    for i in range(num_iters):
        grover_circuit.append(oracle, grover_circuit.qubits)
        grover_circuit.append(diffuser, range(num_of_vars))
    return grover_circuit


def build_diffuser(num_of_vars: int) -> Gate:
    qc = QuantumCircuit(num_of_vars)

    qc.h(qc.qubits)
    qc.x(qc.qubits)
    qc.append(ZGate().control(num_of_vars - 1), range(num_of_vars))
    qc.x(qc.qubits)
    qc.h(qc.qubits)

    U_s = qc.to_gate()
    U_s.name = "Diffuser"
    return U_s


def solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend, save_histogram: bool = False, histogram_title: str = "histogram") -> list[dict]:

    # sort the proposition atoms
    cnf_atoms = sorted(logical_formula.atoms(), key=lambda x: x.name)
    nb_vars = len(cnf_atoms)

    nb_solution = 2
    nb_iter = floor(pi/4 * sqrt((2**nb_vars)/nb_solution))

    grover_circuit = build_grover_circuit(logical_formula_to_oracle, nb_vars, nb_iter)

    # measurement
    grover_circuit.measure(range(nb_vars), range(nb_vars))

    # Simulate and plot results
    transpiled_qc = transpile(grover_circuit, backend)
    job = backend.run(transpiled_qc)
    results = job.result().get_counts()

    # convert results in an easy to understand format
    boolean_solutions = utils.quantum_results_to_boolean(results, cnf_atoms)

    if save_histogram:
        utils.save_histogram_png(results, histogram_title)

    return boolean_solutions


def validate_grover_solutions(results: list[dict], cnf: And):
    for result in results:
        print(cnf.subs(result))
