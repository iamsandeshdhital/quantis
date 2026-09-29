#!/usr/bin/env python3
"""Run all QUANTIS experiments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qit.error_correction.surface_code import create_surface_code
from qit.algorithms.circuit_optimizer import QuantumCircuit, Gate, CircuitOptimizer
from qit.ml.quantum_kernel import create_quantum_kernel, VariationalQuantumClassifier

import numpy as np


def main():
    print("QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems")
    print("Running all experiments...\n")

    # 1. Surface code
    print("[1] Surface Code")
    code = create_surface_code(distance=3)
    sim = code.simulate_error(error_rate=0.01, num_trials=100)
    print(f"    Success rate: {sim['success_rate']:.2%}\n")

    # 2. Circuit optimization
    print("[2] Circuit Optimization")
    circuit = QuantumCircuit(num_qubits=4, name="experiment")
    circuit.add_gate(Gate("H", [0]))
    circuit.add_gate(Gate("X", [1]))
    circuit.add_gate(Gate("H", [0]))

    optimizer = CircuitOptimizer()
    optimized = optimizer.optimize(circuit)
    print(f"    Gates: {circuit.gate_count()} -> {optimized.gate_count()}\n")

    # 3. Quantum ML
    print("[3] Quantum ML")
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = np.sign(X[:, 0] + X[:, 1])

    vqc = VariationalQuantumClassifier(num_qubits=4, max_iterations=20)
    result = vqc.fit(X, y)
    predictions = vqc.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"    Accuracy: {accuracy:.2f}")
    print(f"    Final loss: {result['final_loss']:.4f}\n")

    print("All experiments complete!")


if __name__ == "__main__":
    main()
