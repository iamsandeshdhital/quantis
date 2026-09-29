"""Tests for surface code implementation."""

import pytest
import numpy as np

from qit.error_correction.surface_code import (
    SurfaceCode,
    Stabilizer,
    create_surface_code,
    benchmark_surface_code,
)


class TestStabilizer:
    def test_create(self):
        stab = Stabilizer(pauli_string='XXXX', qubits=[0, 1, 2, 3], stab_type='X')
        assert stab.weight() == 4

    def test_commutes_with(self):
        s1 = Stabilizer(pauli_string='XXXX', qubits=[0, 1, 2, 3], stab_type='X')
        s2 = Stabilizer(pauli_string='ZZZZ', qubits=[0, 1, 2, 3], stab_type='Z')
        assert s1.commutes_with(s2)

    def test_weight(self):
        stab = Stabilizer(pauli_string='XIXI', qubits=[0, 1, 2, 3], stab_type='X')
        assert stab.weight() == 2


class TestSurfaceCode:
    def test_create_d3(self):
        code = create_surface_code(distance=3)
        assert code.distance == 3
        assert code.num_data_qubits == 9
        assert code.num_ancilla_qubits == 4

    def test_create_d5(self):
        code = create_surface_code(distance=5)
        assert code.distance == 5
        assert code.num_data_qubits == 25
        assert code.num_ancilla_qubits == 16

    def test_invalid_distance(self):
        with pytest.raises(ValueError):
            create_surface_code(distance=2)
        with pytest.raises(ValueError):
            create_surface_code(distance=4)

    def test_stabilizers(self):
        code = create_surface_code(distance=3)
        assert len(code.stabilizers) == 8

    def test_stabilizer_types(self):
        code = create_surface_code(distance=3)
        x_stabs = [s for s in code.stabilizers if s.stab_type == 'X']
        z_stabs = [s for s in code.stabilizers if s.stab_type == 'Z']
        assert len(x_stabs) == 4
        assert len(z_stabs) == 4

    def test_syndrome_extraction(self):
        code = create_surface_code(distance=3)
        circuit = code.syndrome_extraction_circuit()
        assert circuit["num_qubits"] == 13
        assert circuit["num_stabilizers"] == 8

    def test_simulate_error(self):
        code = create_surface_code(distance=3)
        result = code.simulate_error(error_rate=0.01, num_trials=100)
        assert "success_rate" in result
        assert "logical_error_rate" in result
        assert 0 <= result["success_rate"] <= 1

    def test_code_parameters(self):
        code = create_surface_code(distance=3)
        params = code.code_parameters()
        assert params["distance"] == 3
        assert params["num_data_qubits"] == 9
        assert params["error_threshold"] == 0.01


class TestBenchmarkSurfaceCode:
    def test_benchmark(self):
        results = benchmark_surface_code(distance=3)
        assert "distance" in results
        assert "num_qubits" in results
        assert "simulation_results" in results
