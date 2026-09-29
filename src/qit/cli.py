"""Command-line interface for QUANTIS Framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .error_correction.surface_code import create_surface_code, benchmark_surface_code
from .algorithms.circuit_optimizer import (
    Gate, QuantumCircuit, CircuitOptimizer, benchmark_optimizer,
    create_linear_coupling_map, CouplingMap,
)
from .ml.quantum_kernel import (
    create_quantum_kernel, benchmark_quantum_ml,
    VariationalQuantumClassifier, QuantumFeatureMap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qit-run",
        description="QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"qit {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Surface code command
    surface_parser = subparsers.add_parser("surface", help="Surface code operations")
    surface_parser.add_argument("--distance", type=int, default=3)
    surface_parser.add_argument("--benchmark", action="store_true")
    surface_parser.add_argument("--simulate", action="store_true")
    surface_parser.add_argument("--error-rate", type=float, default=0.01)

    # Optimizer command
    optimizer_parser = subparsers.add_parser("optimize", help="Circuit optimization")
    optimizer_parser.add_argument("--benchmark", action="store_true")
    optimizer_parser.add_argument("--route", action="store_true")

    # ML command
    ml_parser = subparsers.add_parser("ml", help="Quantum ML")
    ml_parser.add_argument("--benchmark", action="store_true")
    ml_parser.add_argument("--samples", type=int, default=100)
    ml_parser.add_argument("--train", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run full demo")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "surface":
        if args.benchmark:
            print(f"Benchmarking surface code (distance={args.distance})...")
            results = benchmark_surface_code(distance=args.distance)
            print(json.dumps(results, indent=2, default=str))
        elif args.simulate:
            print(f"Simulating errors (distance={args.distance}, error_rate={args.error_rate})...")
            code = create_surface_code(distance=args.distance)
            results = code.simulate_error(error_rate=args.error_rate, num_trials=1000)
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"Creating surface code (distance={args.distance})...")
            code = create_surface_code(distance=args.distance)
            params = code.code_parameters()
            print(json.dumps(params, indent=2))

    elif args.command == "optimize":
        if args.benchmark:
            print("Benchmarking circuit optimizer...")
            circuit = QuantumCircuit(num_qubits=4, name="sample")
            circuit.add_gate(Gate("H", [0]))
            circuit.add_gate(Gate("X", [1]))
            circuit.add_gate(Gate("H", [0]))
            circuit.add_gate(Gate("CNOT", [0, 1]))
            circuit.add_gate(Gate("CNOT", [0, 1]))
            circuit.add_gate(Gate("RZ", [2], [0.5]))
            circuit.add_gate(Gate("RZ", [2], [-0.5]))

            results = benchmark_optimizer(circuit)
            print(json.dumps(results, indent=2, default=str))
        elif args.route:
            print("Routing circuit on linear coupling map...")
            coupling = create_linear_coupling_map(4)
            optimizer = CircuitOptimizer(coupling_map=coupling)

            circuit = QuantumCircuit(num_qubits=4, name="to_route")
            circuit.add_gate(Gate("H", [0]))
            circuit.add_gate(Gate("CNOT", [0, 3]))  # Non-adjacent

            routed = optimizer.route_circuit(circuit)
            print(f"  Original gates: {circuit.gate_count()}")
            print(f"  Routed gates: {routed.gate_count()}")
            print(f"  SWAPs inserted: {routed.gate_count() - circuit.gate_count()}")

    elif args.command == "ml":
        if args.benchmark:
            print(f"Benchmarking quantum ML (samples={args.samples})...")
            results = benchmark_quantum_ml(n_samples=args.samples)
            print(json.dumps(results, indent=2, default=str))
        elif args.train:
            print("Training variational quantum classifier...")
            np.random.seed(42)
            X = np.random.randn(50, 4)
            y = np.sign(X[:, 0] + X[:, 1])

            vqc = VariationalQuantumClassifier(num_qubits=4, max_iterations=50)
            result = vqc.fit(X, y)
            predictions = vqc.predict(X)
            accuracy = np.mean(predictions == y)

            print(f"  Final loss: {result['final_loss']:.4f}")
            print(f"  Accuracy: {accuracy:.2f}")

    elif args.command == "demo":
        print("=" * 60)
        print("QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems")
        print("=" * 60)

        # 1. Surface code
        print("\n[1] Surface Code (distance=3)")
        code = create_surface_code(distance=3)
        params = code.code_parameters()
        print(f"    Data qubits: {params['num_data_qubits']}")
        print(f"    Ancilla qubits: {params['num_ancilla_qubits']}")
        print(f"    Stabilizers: {params['num_stabilizers']}")

        # Error simulation
        sim = code.simulate_error(error_rate=0.01, num_trials=100)
        print(f"    Error simulation (p=0.01):")
        print(f"      Success rate: {sim['success_rate']:.2%}")
        print(f"      Logical error rate: {sim['logical_error_rate']:.2%}")

        # 2. Circuit optimization
        print("\n[2] Circuit Optimization")
        circuit = QuantumCircuit(num_qubits=4, name="demo")
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("X", [1]))
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("CNOT", [0, 1]))
        circuit.add_gate(Gate("CNOT", [0, 1]))

        optimizer = CircuitOptimizer()
        optimized = optimizer.optimize(circuit)
        print(f"    Original gates: {circuit.gate_count()}")
        print(f"    Optimized gates: {optimized.gate_count()}")
        print(f"    Reduction: {(1 - optimized.gate_count() / circuit.gate_count()) * 100:.1f}%")

        # 3. Quantum ML
        print("\n[3] Quantum ML")
        results = benchmark_quantum_ml(n_samples=30)
        print(f"    VQC accuracy: {results['vqc']['accuracy']:.2f}")
        print(f"    Final loss: {results['vqc']['final_loss']:.4f}")

        print("\n" + "=" * 60)
        print("Demo complete!")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
