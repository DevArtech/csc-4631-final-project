# Library imports
import math
import random
import torch
import pennylane as qml

def quantum_circuit(noise, weights, n_qubits, q_depth):
    """Quantum circuit function. Creates device dynamically based on n_qubits."""
    # Create device dynamically
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def _circuit(noise, weights, n_qubits, q_depth):
        weights = weights.reshape(q_depth, n_qubits)

        # Initialise latent vectors
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        # Repeated layer
        for i in range(q_depth):
            # Parameterised layer
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)

            # Control Z gates
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(n_qubits)))
    
    return _circuit(noise, weights, n_qubits, q_depth)


# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights, n_qubits, q_depth, n_a_qubits):
    probs = quantum_circuit(noise, weights, n_qubits, q_depth)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven
