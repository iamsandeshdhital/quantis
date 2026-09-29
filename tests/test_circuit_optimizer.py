"""Tests for circuit optimizer."""

import pytest

from qit.algorithms.circuit_optimizer import (
    Gate,
    QuantumCircuit,
    CircuitOptimizer,
    CouplingMap,
    estimate_resources,
    benchmark_optimizer,
    create_linear_coupling_map,
    create_grid_coupling_map,
)


class TestGate:
    def test_create(self):
        gate = Gate("H", [0])
        assert gate.name == "H"
        assert gate.qubits == [0]

    def test_repr(self):
        gate = Gate("RX", [0], [0.5])
        assert "RX" in repr(gate)


class TestQuantumCircuit:
    def test_create(self):
        circuit = QuantumCircuit(num_qubits=4)
        assert circuit.num_qubits == 4
        assert circuit.gate_count() == 0

    def test_add_gate(self):
        circuit = QuantumCircuit(num_qubits=4)
        circuit.add_gate(Gate("H", [0]))
        assert circuit.gate_count() == 1

    def test_depth(self):
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("CNOT", [0, 1]))
        assert circuit.depth() == 2

    def test_two_qubit_gate_count(self):
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("CNOT", [0, 1]))
        assert circuit.two_qubit_gate_count() == 1


class TestCouplingMap:
    def test_create_linear(self):
        cmap = create_linear_coupling_map(4)
        assert cmap.num_qubits == 4
        assert len(cmap.edges) == 3

    def test_create_grid(self):
        cmap = create_grid_coupling_map(2, 2)
        assert cmap.num_qubits == 4
        assert len(cmap.edges) == 4

    def test_shortest_path(self):
        cmap = create_linear_coupling_map(4)
        path = cmap.shortest_path(0, 3)
        assert path == [0, 1, 2, 3]

    def test_distance(self):
        cmap = create_linear_coupling_map(4)
        assert cmap.distance(0, 1) == 1
        assert cmap.distance(0, 3) == 3


class TestCircuitOptimizer:
    def test_cancel_inverse_gates(self):
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(Gate("X", [0]))
        circuit.add_gate(Gate("X", [0]))

        optimizer = CircuitOptimizer()
        optimized = optimizer.cancel_inverse_gates(circuit)
        assert optimized.gate_count() == 0

    def test_merge_rotations(self):
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(Gate("RZ", [0], [0.5]))
        circuit.add_gate(Gate("RZ", [0], [0.3]))

        optimizer = CircuitOptimizer()
        optimized = optimizer.merge_rotations(circuit)
        assert optimized.gate_count() == 1
        assert abs(optimized.gates[0].params[0] - 0.8) < 1e-10

    def test_optimize(self):
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("X", [1]))
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("CNOT", [0, 1]))
        circuit.add_gate(Gate("CNOT", [0, 1]))

        optimizer = CircuitOptimizer()
        optimized = optimizer.optimize(circuit)
        assert optimized.gate_count() <= circuit.gate_count()

    def test_route_circuit(self):
        coupling = create_linear_coupling_map(4)
        optimizer = CircuitOptimizer(coupling_map=coupling)

        circuit = QuantumCircuit(num_qubits=4)
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("CNOT", [0, 3]))

        routed = optimizer.route_circuit(circuit)
        assert routed.gate_count() > circuit.gate_count()  # SWAPs inserted


class TestEstimateResources:
    def test_estimate(self):
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("CNOT", [0, 1]))

        resources = estimate_resources(circuit)
        assert resources["num_qubits"] == 2
        assert resources["gate_count"] == 2
        assert resources["two_qubit_gate_count"] == 1


class TestBenchmarkOptimizer:
    def test_benchmark(self):
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(Gate("H", [0]))
        circuit.add_gate(Gate("X", [1]))
        circuit.add_gate(Gate("H", [0]))

        results = benchmark_optimizer(circuit)
        assert "original" in results
        assert "optimized" in results
        assert "optimization_time_s" in results
