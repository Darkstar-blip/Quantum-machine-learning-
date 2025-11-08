#!/usr/bin/env python3
import sys
try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    print("IMPORT_OK: QSVC and FidelityQuantumKernel available")
    sys.exit(0)
except Exception as e:
    print("IMPORT_FAIL2", e)
    sys.exit(2)
