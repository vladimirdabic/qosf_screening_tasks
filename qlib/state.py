from typing import List, Tuple
import numpy as np


class State:
    matrix: np.ndarray
    qubits: int

    def __init__(self, amplitudes: List[complex] = None, num_qubits: int = None):
        if amplitudes is not None:
            self.matrix = self.__prepare_state(amplitudes)
        elif num_qubits is not None:
            self.qubits = num_qubits
            state_vector = np.zeros(2 ** num_qubits, dtype=np.complex128)
            state_vector[0] = 1 # |00 .. 0>
            self.matrix = np.outer(state_vector, state_vector.conj())
        else:
            raise ValueError("Must provide amplitudes or num_qubits")

    def __prepare_state(self, amplitudes: List[complex]):
        state = np.array(amplitudes, dtype=np.complex128)

        # Normalize if sum is not 1
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0):
            state = state / norm

        # Check dimension (must be 2^n)
        n_qubits = int(np.log2(len(state)))
        if 2**n_qubits != len(state):
            raise ValueError("Number of amplitudes must be 2^n for n qubits")
        
        self.qubits = n_qubits

        # Create a density matrix
        state2d = state[np.newaxis]
        col_vec = state2d.T
        return col_vec @ state2d.conj()

    def apply_gate(self, U: np.ndarray, qubits: Tuple[int]):
        I = np.eye(2, dtype=np.complex128)

        # I I ... U ... I
        operators = [I] * self.qubits
        for qubit_idx in qubits:
            operators[qubit_idx] = U

        full_U = operators[0]
        for i in range(1, self.qubits):
            full_U = np.kron(full_U, operators[i])

        self.matrix = full_U @ self.matrix @ full_U.conj().T

    def apply_operator(self, U: np.ndarray):
        self.matrix = U @ self.matrix @ U.conj().T

    def expectation_value(self, U: np.ndarray):
        return np.trace(U @ self.matrix)
    
    def probabilities(self):
        return np.abs(self.matrix.diagonal())