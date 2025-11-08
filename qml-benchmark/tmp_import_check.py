#!/usr/bin/env python3
import sys
try:
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    print("IMPORT_OK:modern")
    sys.exit(0)
except Exception as e:
    try:
        from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
        from qiskit_machine_learning.algorithms.classifiers import QSVC
        print("IMPORT_OK:old")
        sys.exit(0)
    except Exception as e2:
        print("IMPORT_FAIL", e, e2)
        sys.exit(2)
