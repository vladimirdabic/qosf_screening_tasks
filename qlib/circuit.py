from __future__ import annotations
from typing import List, Tuple, Iterable
from abc import ABC
from .state import State
from dataclasses import dataclass
import numpy as np
from . import ops


@dataclass
class CircuitEntry:
    is_operator: bool   # if true, then it acts on the whole state, if false, it acts on a singular qubit
    matrix: np.ndarray
    qubits: Tuple[int]
    noise_model: NoiseModel

class Circuit:
    gates: List[CircuitEntry]
    qubits: int

    def __init__(self, num_qubits: int = 1):
        self.gates = []
        self.qubits = num_qubits

    def add_gate(self, gate: np.ndarray, qubit: int = None, qubits: Tuple[int] = None, noise_model: NoiseModel = None):
        if qubit is not None:
            self.gates.append(CircuitEntry(False, gate, (qubit,), noise_model))
        elif qubits is not None:
            self.gates.append(CircuitEntry(False, gate, qubits, noise_model))
        else:
            raise ValueError("Expected target qubit(s)")
        
    def add_controlled_gate(self, gate: np.ndarray, control_qubit: int, target_qubit: int, noise_model: NoiseModel = None):
        I = np.eye(2, dtype=np.complex128)

        ops_0 = [I] * self.qubits
        ops_0[control_qubit] = ops.P_0

        ops_1 = [I] * self.qubits
        ops_1[control_qubit] = ops.P_1
        ops_1[target_qubit] = gate 

        op_0 = self.__build_operator(ops_0)
        op_1 = self.__build_operator(ops_1)
        cgate = op_0 + op_1

        self.add_operator(cgate, noise_model=noise_model)
        
    def add_operator(self, U: np.ndarray, noise_model: NoiseModel = None):
        self.gates.append(CircuitEntry(True, U, None, noise_model))
        
    def run(self, initial_state: State = None, noise_model: NoiseModel = None) -> State:
        state = initial_state or State(num_qubits = self.qubits)
        
        for i, circuit_entry in enumerate(self.gates):
            if circuit_entry.is_operator:
                state.apply_operator(circuit_entry.matrix)
            else:
                state.apply_gate(circuit_entry.matrix, circuit_entry.qubits)
            
            # Apply gate-specific noise
            if circuit_entry.noise_model:
               circuit_entry.noise_model.apply(state)
            
            # Apply global noise
            if noise_model:
                noise_model.apply(state)
        
        return state
    
    def CNOT(self, control_qubit: int, target_qubit: int, noise_model: NoiseModel = None):
        self.add_controlled_gate(ops.X, control_qubit, target_qubit, noise_model=noise_model)

    def __build_operator(self, operators: List[np.ndarray]) -> np.ndarray:
        full_U = operators[0]
        for i in range(1, self.qubits):
            full_U = np.kron(full_U, operators[i])

        return full_U

class NoiseModel(ABC):
    def apply(self, state: State):
        raise NotImplementedError("Apply method for noise model not implemented")
    

def build_operator(rho: np.ndarray, qubit_index: int, U: np.ndarray) -> np.ndarray:
    num_qubits = int(np.log2(rho.shape[0]))
    I = np.eye(2, dtype=np.complex128)
    
    # I I ... U ... I
    operators = [I] * num_qubits
    operators[qubit_index] = U

    full_U = operators[0]
    for i in range(1, num_qubits):
        full_U = np.kron(full_U, operators[i])

    return full_U

def build_operators(rho: np.ndarray, qubits: Iterable[int], operator_list: List[np.ndarray]) -> np.ndarray:
    num_qubits = int(np.log2(rho.shape[0]))
    I = np.eye(2, dtype=np.complex128)
    
    # I I ... U ... I
    operators = [I] * num_qubits

    for i, qubit in enumerate(qubits):
        operators[qubit] = operator_list[i]

    full_U = operators[0]
    for i in range(1, num_qubits):
        full_U = np.kron(full_U, operators[i])

    return full_U

def repeat_operator(operator: np.ndarray, qubits: int) -> np.ndarray:
    operators = [operator] * qubits

    full_U = operators[0]
    for i in range(1, qubits):
        full_U = np.kron(full_U, operators[i])

    return full_U