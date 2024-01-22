##########################################################################

# Titre: GroverUtils.py
# Author: Christopher Sicotte (SICC2201)
# last modified: 21/01/2024

##########################################################################
'''
# Description:

This file contains all the methods that we need to run a full Grover algorithm given logical formulas.


# Methods:

- args_to_toffoli(qc: QuantumCircuit, variables: list[symbols],  proposition, index: int) : Build toffoli gates from propositions to append in the global quantum circuit.

- cnf_to_oracle(logical_formula: And) -> Gate : Translates a normal conjunctive logical formula into an oracle that takes the form of a quantum gate.

- build_diffuser(num_of_vars: int) -> Gate : Build the diffuser depending on the number of input qubits.

- build_grover_circuit(oracle: Gate, cnf: And, num_iters: int) -> Gate : Build the Grover algorithm from the inputs.

- solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend: Backend) ->  dict{string:bool}: Given a logical formula, converts
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

"""Commentaires
Une porte Toffoli est spécifique au CCX. 
Ici il s'agit de multi-contrôle avec une nombre arbitraire de qubits de control.

Le problème de Pincus et du Cake devrait avoir la même forme. Pas de if nécessaire.

Je te suggère de rendre cette fonction spécifique à une disjonction.
Donc le nom de la fonction est à changer. Suggestion : "disjonction_t0_gate"?

C'est une mauvaise pratique de modifier un argument de la fonction sauf si vraiment plus performant.
Ici je te suggère de retourner la porte et qubit_index et l'ajouter au circuit dans cnf_to_oracle.
"""
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
    toffoli_gate = XGate().control(len(toffoli_qubits), ctrl_state = toffoli_qubits[::-1]) # [::-1] inverse the string to respect little endian
    qc.append(toffoli_gate, qubit_index)

"""Commentaires
Type hint pour le retour.
Cette fonction ne retourne pas vraiment l'oracle, mais seulement la première partie avant le MCZ.
Je garderais cette fonction avec un autre nom.
Je ferrais une nouvelle fonction cnf_to_oracle qui appelle celle-ci, ajoute le MCZ et applique l'inverse.
Voir les commentaires de la fonction précédente.
Enlève les print et le draw.
Les commentaires ne sont pas indispensables. Le code se comprend assez bien.
Tu peut spécifier le nom de ta gate avec `oracle_gate = qc.to_gate(name = "Oracle")`
"""
def cnf_to_oracle(logical_formula: And):

    print(logical_formula)
    # sort the proposition atoms
    variables = sorted(logical_formula.atoms(), key=lambda x: x.name)
    # print("proposition values: ", variables, " of type: ", type(variables),  " of lenght: ", len(variables))

    # create a circuit with 2 registers
    variables_circuit = QuantumRegister(len(variables), "var_qubits")
    clauses_circuit = QuantumRegister(len(logical_formula.args), "anc_qubits")
    qc = QuantumCircuit(variables_circuit, clauses_circuit)

    # Apply the right toffoli gate to the circuit
    i = len(variables)
    for clause in logical_formula.args:
        print("clause", clause)
        args_to_toffoli(qc, list(variables),  clause, i) 

        if isinstance(clause, Or): # if the proposition is a OR, add an X gate
            qc.x(i)

        print("*********************  oracle part " + str(i - len(variables)) + " ************************")
        i += 1

    qc.draw("mpl")

    # create the oracle gate
    oracle_gate = qc.to_gate()
    oracle_gate.name = "Oracle"
    return oracle_gate

"""Commentaires
Il manque des types hints.
Type hint pour le retour.
Je ne suis pas sûr de comprendre ton commentaire.
Tu dois fournir le nombre de qubit de variable pour savoir la taille du diffuseur.
L'utilisation de MCMT n'est pas nécessaire et ce type de porte est exponentiellement couteuse.
Utilise plus tôt MCZ = ZGate().control(...).
Comme dit plus tôt, l'application du MCZ devrait être dans cnf_to_oracle.
Évite d'appeler `build_diffuser` à chaque itération. Une fois au début en ensuite append. Même chose pour l'oracle.
"""
# in this methods i have a "cnf" in parameter because, for pincus, the size of the atoms != the size of the clauses, so i need to differentiate them in my registers
def build_grover_circuit(gate, cnf, num_iters: int):
    
    num_of_vars = len(cnf.atoms())

    # create the global circuit
    variables_circuit = QuantumRegister(num_of_vars, name = "variables")
    clauses_circuit = QuantumRegister(len(cnf.args), name = "clauses")
    cr = ClassicalRegister(num_of_vars, name = "CR")
    qc = QuantumCircuit(variables_circuit, clauses_circuit, cr)

    # initialize circuit |s> state
    grover_circuit = utils.initialize_s(qc, 0, variables_circuit.size) 

    # apply num_iters times the oracle, a mulicontrolled-Z gate, the inversse oracle and the diffuser
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

"""Commentaires
Même commentaire pour le MCZ, utilise ZGate().control().
Type hint pour le retour.
"""
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

    # create the gate
    U_s = qc.to_gate()
    U_s.name = "Diffuser"
    return U_s

"""Comentaires 
Il manque le type hint pour backend et le type hint pour le retour.
À quoi sert `nb_solution`?
Je ne crois pas que ce soit nécessaire d'avoir une fonction `mesure_qubits` dans utils. Je verrai quand je regarderai ce fichier.
Pas de print et pas pas afficher d'histogramme ici. 
Une manière de pouvoir faire un histogramme avec une bonne structure serait d'avoir une fonction qui execute le circuit grover
et extrait les résultats. Cette fonction serait appelée dans `solve_sat_with_grover`. L'utilisateur pourrait utiliser séparément
pour générer un histogramme.
La method `quantum_results_to_boolean` semble assez spécifique à la solution de SAT avec grover pour se retrouver ici, mais je peux me tromper.
Je vais aller voir si cette fonction retourne le bon type.
"""
def solve_sat_with_grover(logical_formula: And, logical_formula_to_oracle: Callable, backend):

    nb_solution = 1
    nb_qubits = len(logical_formula.atoms())

    # ******************************************************
    # the ideal num of iterations is supposeed to be
    # nb_iter = floor(pi/4 * sqrt(nb_qubits/nb_solution))
    # but it does not work. So in the meantime i harcoded 2 since its the best.
    nb_iter = 2
    # *******************************************************

    print("num_iterations = ", nb_iter)
    # sort the proposition atoms
    cnf_atoms = sorted(logical_formula.atoms(), key=lambda x: x.name) 

    # build quantum circuit
    grover_circuit = build_grover_circuit(logical_formula_to_oracle, logical_formula, nb_iter)

    # measurement
    utils.mesure_qubits(grover_circuit, len(logical_formula.atoms()))

    # Simulate and plot results
    transpiled_qc = transpile(grover_circuit, backend)
    job = backend.run(transpiled_qc)
    results = list(job.result().get_counts().items())
    plot_histogram(job.result().get_counts())

    # print(results)

    # convert results in an easy to understand format
    boolean_solutions = utils.quantum_results_to_boolean(results, cnf_atoms)

    print(boolean_solutions)

    return boolean_solutions


