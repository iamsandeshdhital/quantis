#!/usr/bin/env python3
"""Run all QUANTIS benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qit.error_correction.surface_code import benchmark_surface_code
from qit.algorithms.circuit_optimizer import QuantumCircuit, Gate, benchmark_optimizer
from qit.ml.quantum_kernel import benchmark_quantum_ml


def main():
    print("=" * 60)
    print("QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems")
    print("Benchmark Suite")
    print("=" * 60)

    # Surface code benchmark
    print("\n[1] Surface Code Benchmark")
    result = benchmark_surface_code(distance=3)
    print(f"    Distance: {result['distance']}")
    print(f"    Qubits: {result['num_qubits']}")
    print(f"    Success rate: {result['simulation_results']['success_rate']:.2%}")

    # Circuit optimizer benchmark
    print("\n[2] Circuit Optimizer Benchmark")
    circuit = QuantumCircuit(num_qubits=4, name="benchmark")
    circuit.add_gate(Gate("H", [0]))
    circuit.add_gate(Gate("X", [1]))
    circuit.add_gate(Gate("H", [0]))
    circuit.add_gate(Gate("CNOT", [0, 1]))
    circuit.add_gate(Gate("CNOT", [0, 1]))

    result = benchmark_optimizer(circuit)
    print(f"    Original gates: {result['original']['gate_count']}")
    print(f"    Optimized gates: {result['optimized']['gate_count']}")
    print(f"    Reduction: {result['gate_reduction_percent']:.1f}%")

    # Quantum ML benchmark
    print("\n[3] Quantum ML Benchmark")
    result = benchmark_quantum_ml(n_samples=30)
    print(f"    VQC accuracy: {result['vqc']['accuracy']:.2f}")
    print(f"    Final loss: {result['vqc']['final_loss']:.4f}")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")


if __name__ == "__main__":
    main()
