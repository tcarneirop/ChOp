# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:29:08 2026

@author: Jérôme
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:36:03 2026

@author: Jérôme
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import routing    ### C++ routing by JP

import argparse
parser = argparse.ArgumentParser()

##-------------------------------------------------------
##      Utility function
##-------------------------------------------------------
def get_circuit_array(circuit):
    """
    Returns the circuit's gate sequence as a (num_gates, 2) numpy int32 array.
    For each gate, column 0 holds the first qubit i  print(f"Mapping = {mapping}\nDepth using C++ = {depth_cpp}\nDepth using real function = {depthOG}")
NameError: name 'depth_cpp' is not defndex and column 1 holds
    the second qubit index (or -1 for single-qubit gates).

    Parameters:
        circuit (qiskit.circuit.quantumcircuit.QuantumCircuit): A quantum circuit.

    Return:
        numpy.ndarray : (num_gates, 2) array of qubit indices, C-contiguous.
    """
    qubit_to_idx = {q: i for i, q in enumerate(circuit.qubits)}
    data = circuit.data
    gates = np.empty((len(data), 2), dtype=np.int32)
    for i, inst in enumerate(data):
        qubits = inst.qubits
        gates[i, 0] = qubit_to_idx[qubits[0]]
        gates[i, 1] = qubit_to_idx[qubits[1]] if len(qubits) == 2 else -1
    return gates

##-------------------------------------------------------
##      Fitness functions
##-------------------------------------------------------
### Fast C++ function
def fitness_cpp(circuit, mapping, Dist):
    qc_gates = get_circuit_array(circuit)
    depth_batch, num_gates_batch = routing.circuit_routing_batch(qc_gates, Dist, [mapping])
    return depth_batch[0]

### Slow but slightly more accurate function
def fitnessOG(qc, layout, backend):
    qr=qc.qregs[0]
    init_layout={qr[i]:layout[i] for i in range(len(layout))}
    init_layout=Layout(init_layout)

    pm = generate_preset_pass_manager(3,backend,initial_layout=init_layout)
    pm.layout.remove(1)
    pm.layout.remove(1)

    QC=pm.run(qc)
    return QC.depth()

##-------------------------------------------------------
##      Circuit selector
##-------------------------------------------------------
def circuit_selector(name,nb_qubit):
    if name=="ghzall":
        qc=QuantumCircuit(nb_qubit)
        qc.h(0)
        for i in range(1,nb_qubit):
            qc.cx(0,i)
        qc.measure_all()
          
        qr=qc.qregs[0]

        name="ghzall"
    else:
        filename=f"{name}_indep_qiskit_{nb_qubit}"
        qasmfile=f"../Benchmarks/MQTBench_all/{filename}.qasm"
        qc=QuantumCircuit().from_qasm_file(qasmfile)
        qr=qc.qregs[0]
    return qc,qr

parser.add_argument("-c", "--circuit", nargs=2, default=["ising_model_16", "16"], help="Name and number of qubits of the circuit")
parser.add_argument("-b", "--backend", default=["JP"], choices=["OG","JP"], help="Type of Backend")
args = parser.parse_args()

##-------------------------------------------------------
##      Circuit & Backend selection
##-------------------------------------------------------
if args.backend == "OG":
    ### Backend
    from qiskit.providers.fake_provider import Fake127QPulseV1, Fake5QV1, Fake20QV1, Fake27QPulseV1# New import for the backend
    from qiskit.providers import BackendV2Converter
    
    backend = Fake127QPulseV1()
    backend = BackendV2Converter(backend)
    
    ##### Circuit (OG)
    circuit_name = args.circuit[0]
    nb_qubit = int(args.circuit[1])
    
    circuit,qr=circuit_selector(circuit_name, nb_qubit)
    circuit.remove_final_measurements() ## Important, else bugs
    
elif args.backend == "JP":
    ##### JP Backend
    from qiskit.transpiler import CouplingMap
    
    coupling = CouplingMap([
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13),
        (13,14), (14,15), (0,15), (1,14), (2,13), (3,12), (4,11), (5,10), (6,9)
    ])
    
    from qiskit.transpiler import Target
    from qiskit.circuit.library import IGate, XGate, CXGate, RZGate, SXGate, Reset
    from qiskit.circuit import Measure
    
    target = Target(num_qubits=16)
    
    # 1-qubit gates
    target.add_instruction(IGate(),   {(q,): None for q in range(target.num_qubits)})
    target.add_instruction(XGate(),   {(q,): None for q in range(target.num_qubits)})
    target.add_instruction(SXGate(),  {(q,): None for q in range(target.num_qubits)})
    target.add_instruction(RZGate(0), {(q,): None for q in range(target.num_qubits)})
    target.add_instruction(Reset(),   {(q,): None for q in range(target.num_qubits)})
    target.add_instruction(Measure(), {(q,): None for q in range(target.num_qubits)})
    
    # 2-qubit gates follow coupling graph
    cx_edges = set(coupling.get_edges())
    cx_edges |= {(b, a) for (a, b) in coupling.get_edges()}
    target.add_instruction(CXGate(), {edge: None for edge in cx_edges})
        
    from qiskit.providers import BackendV2
    from qiskit.providers.options import Options
    from qiskit.providers.exceptions import QiskitError
    
    class CustomBackend(BackendV2):
        def __init__(self, target, coupling):
            super().__init__(name="custom_basis_backend")
            self._target = target
            self._coupling = coupling
    
        @property
        def target(self):
            return self._target
    
        @property
        def coupling_map(self):
            return self._coupling
    
        @classmethod
        def _default_options(cls):
            return Options(shots=1024)
        
        @property
        def max_circuits(self):
            return 1
        
        def run(self, run_input, **options):
            raise QiskitError(
                "This backend is for transpilation only and does not support execution."
            )
    backend = CustomBackend(target, coupling)
    
    ##### Circuit (JP)
    folder = "./Benchmark_Small/"
    # file = "ising_model_16.qasm"
    file = args.circuit[0]
    circuit_name = file.split('.')[0]  # name without extension
    circuit=QuantumCircuit().from_qasm_file(folder+file+'.qasm')
    circuit.remove_final_measurements() ## Important, else bugs
    qr=circuit.qregs[0]
    nb_qubit = circuit.num_qubits


##-------------------------------------------------------
##      Testing area 
##-------------------------------------------------------
Dist = backend.coupling_map.distance_matrix
qc_gates = get_circuit_array(circuit)

# Format the array into a string block and strip any trailing whitespace/newlines
gates_string = "\n".join(f"{row[0]} {row[1]}" for row in qc_gates)

with open(f"circuits/{circuit_name}.txt", "w") as f:
    f.write(f"{nb_qubit}\n")
    f.write(gates_string) # Writes the data with absolutely no trailing newline at the end of the file


### Your mapping here
#mapping = np.random.choice(nb_qubit, backend.num_qubits, replace = False)

#depth_cpp = fitness_cpp(circuit, mapping, Dist)
#depthOG = fitnessOG(circuit, mapping, backend)

device_name = 'Albatroz'
np.savetxt(f"matrices/{device_name}.txt", Dist,fmt='%i')

##### Printing results for testing purposes
#pint(f"Mapping = {mapping}\nDepth using C++ = {depth_cpp}\nDepth using real function = {depthOG}")


