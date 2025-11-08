#!/usr/bin/env python3
import qiskit_machine_learning as qml
import qiskit_machine_learning.kernels as km
import inspect, sys, os
print('qml package version:', getattr(qml, '__version__', 'no-version'))
print('kernels module file:', km.__file__)
print('dir(kernels):', dir(km))
print('\n--- kernels __init__ content ---\n')
print(open(km.__file__, 'r').read())
