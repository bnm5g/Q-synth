from qiskit import QuantumCircuit
from qsynth import get_coupling_graph, layout_synthesis, peephole_synthesis

# 1. Create a 3-qubit circuit with 3 classical bits for measurement
qc = QuantumCircuit(3, 3)
qc.h(0)          
qc.cx(0, 2)      

# Measure qubits 0, 1, and 2 into classical bits 0, 1, and 2
# This avoids adding the auto-generated barrier gate!
qc.measure([0, 1, 2], [0, 1, 2])

# 2. Define your hardware layout
coupling_graph = get_coupling_graph(coupling_graph=[[0, 1], [1, 2]], bidirectional=1)

print("--- Running Layout Synthesis ---")
mapped_result = layout_synthesis(circuit=qc, coupling_graph=coupling_graph)

print("Optimized Circuit Layout:")
print(mapped_result.circuit)

print("\n--- Running Peephole Optimization ---")
opt_result = peephole_synthesis(
    circuit=mapped_result.circuit, 
    coupling_graph=coupling_graph, 
    slicing="cnot", 
    metric="cx-count"
)

print("Peephole Optimization Complete!")
print(opt_result.circuit)