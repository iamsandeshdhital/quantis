# QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems

A revolutionary quantum computing research framework for MSc Quantum
Information Technology students and active research by faculty.

**QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems**
— where quantum information meets advanced nanoscale technology.

## Research Alignment

This project aligns with core research strengths:

| Research Area | Project Component |
|------------------------|-------------------|
| **Formal Methods** | Quantum error correction, verification |
| **Algorithms & Complexity** | Quantum algorithm optimization |
| **AI & Machine Learning** | Quantum machine learning kernels |
| **Software Systems** | Quantum compilation, runtime |
| **Embedded Systems** | Quantum control systems |

## Key Components

### 1. Quantum Error Correction (Formal Methods)
- Surface code implementation
- Stabilizer formalism
- Syndrome extraction
- Decoder optimization (MWPM, union-find)
- Fault-tolerant gate synthesis

### 2. Quantum Algorithm Optimization (Algorithms)
- Circuit optimization passes
- Gate synthesis (KAK decomposition)
- Qubit mapping and routing
- Resource estimation
- Complexity analysis

### 3. Quantum Machine Learning (AI/ML)
- Quantum kernel methods
- Variational quantum circuits
- Quantum data encoding
- Training pipelines
- Benchmarking vs classical ML

### 4. Quantum Compilation (Software Systems)
- Multi-level IR (QASM → LLVM → machine)
- Pass manager
- Backend abstraction
- Profiling and optimization

### 5. Quantum Control (Embedded Systems)
- Pulse-level control
- Calibration routines
- Real-time feedback
- Noise characterization

## Quick Start

```bash
git clone https://github.com/iamsandeshdhital/quantis.git
cd quantis
pip install -e ".[dev]"
python -m qit.cli --demo
```

## Repository Layout

```
├── src/qit/
│   ├── error_correction/    # Surface codes, decoders
│   ├── algorithms/          # Circuit optimization, synthesis
│   ├── ml/                  # Quantum kernels, VQC
│   ├── compilation/         # IR, passes, backends
│   ├── control/             # Pulse control, calibration
│   └── common/              # Shared utilities
├── tests/                   # Comprehensive test suite
├── benchmarks/              # Research benchmarks
├── docs/                    # Documentation
└── examples/                # Example notebooks
```

## Research Papers

This framework implements algorithms from:
- Fowler et al. (2012) - Surface codes
- Bravyi et al. (2018) - Magic state distillation
- Preskill (2018) - NISQ era
- Cerezo et al. (2021) - Variational quantum algorithms

## License

MIT
