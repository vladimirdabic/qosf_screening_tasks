import numpy as np

X = np.array([
    [0, 1],
    [1, 0]
], dtype=np.complex128)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=np.complex128)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=np.complex128)

H = np.array([
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), -1/np.sqrt(2)]
], dtype=np.complex128)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)

P_0 = np.array([
    [1, 0],
    [0, 0]
], dtype=np.complex128)

P_1 = np.array([
    [0, 0],
    [0, 1]
], dtype=np.complex128)