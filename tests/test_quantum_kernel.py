"""Tests for quantum kernel methods."""

import numpy as np
import pytest

from qit.ml.quantum_kernel import (
    QuantumState,
    QuantumFeatureMap,
    QuantumKernel,
    VariationalQuantumClassifier,
    create_quantum_kernel,
    benchmark_quantum_ml,
)


class TestQuantumState:
    def test_create(self):
        state = QuantumState(np.array([1, 0, 0, 0]))
        assert state.num_qubits == 2

    def test_normalization(self):
        state = QuantumState(np.array([2, 0, 0, 0]))
        assert np.isclose(np.linalg.norm(state.amplitudes), 1.0)

    def test_overlap(self):
        s1 = QuantumState(np.array([1, 0, 0, 0]))
        s2 = QuantumState(np.array([1, 0, 0, 0]))
        assert np.isclose(s1.overlap(s2), 1.0)

        s3 = QuantumState(np.array([0, 1, 0, 0]))
        assert np.isclose(s1.overlap(s3), 0.0)


class TestQuantumFeatureMap:
    def test_create(self):
        fm = QuantumFeatureMap(name="zz", num_qubits=4)
        assert fm.num_qubits == 4

    def test_encode_zz(self):
        fm = QuantumFeatureMap(name="zz", num_qubits=2)
        x = np.array([1.0, 2.0])
        state = fm.encode(x)
        assert state.num_qubits == 2
        assert np.isclose(np.linalg.norm(state.amplitudes), 1.0)

    def test_encode_pauli(self):
        fm = QuantumFeatureMap(name="pauli", num_qubits=2)
        x = np.array([1.0, 2.0])
        state = fm.encode(x)
        assert state.num_qubits == 2

    def test_encode_amplitude(self):
        fm = QuantumFeatureMap(name="amplitude", num_qubits=2)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        state = fm.encode(x)
        assert state.num_qubits == 2
        assert np.isclose(np.linalg.norm(state.amplitudes), 1.0)


class TestQuantumKernel:
    def test_create(self):
        qk = create_quantum_kernel(num_qubits=4)
        assert qk.feature_map.num_qubits == 4

    def test_compute(self):
        qk = create_quantum_kernel(num_qubits=2)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        k = qk.compute(x, y)
        assert 0 <= k <= 1

    def test_compute_matrix(self):
        qk = create_quantum_kernel(num_qubits=2)
        X = np.random.randn(5, 2)
        K = qk.compute_matrix(X)
        assert K.shape == (5, 5)
        assert np.allclose(K, K.T)


class TestVariationalQuantumClassifier:
    def test_create(self):
        vqc = VariationalQuantumClassifier(num_qubits=4)
        assert vqc.num_qubits == 4

    def test_predict(self):
        vqc = VariationalQuantumClassifier(num_qubits=2)
        X = np.random.randn(10, 2)
        predictions = vqc.predict(X)
        assert len(predictions) == 10
        assert all(p in [-1, 1] for p in predictions)

    def test_fit(self):
        vqc = VariationalQuantumClassifier(num_qubits=2, max_iterations=5)
        X = np.random.randn(10, 2)
        y = np.sign(X[:, 0])
        result = vqc.fit(X, y)
        assert "final_loss" in result
        assert "history" in result


class TestBenchmarkQuantumML:
    def test_benchmark(self):
        results = benchmark_quantum_ml(n_samples=10)
        assert "quantum_kernel" in results
        assert "vqc" in results
