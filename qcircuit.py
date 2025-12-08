# Library imports
import torch
import pennylane as qml

def quantum_circuit(noise, class_angles, weights, n_qubits, q_depth):
    """
    Efficient conditional quantum circuit.
    Class conditioning applied once at start, then parameterized layers.
    
    Args:
        noise: Latent noise vector of shape (n_qubits,)
        class_angles: Class conditioning angles of shape (n_qubits * 2,) - for RX and RZ
        weights: Trainable parameters of shape (q_depth * n_qubits,)
        n_qubits: Number of qubits
        q_depth: Depth of the circuit
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def _circuit(noise, class_angles, weights, n_qubits, q_depth):
        weights = weights.reshape(q_depth, n_qubits)
        class_rx = class_angles[:n_qubits]
        class_rz = class_angles[n_qubits:]

        # Initial encoding: noise + class conditioning (applied once)
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
            qml.RX(class_rx[i], wires=i)
            qml.RZ(class_rz[i], wires=i)

        # Parameterized layers with entanglement
        for layer in range(q_depth):
            # Entanglement
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
            
            # Parameterised rotations
            for y in range(n_qubits):
                qml.RY(weights[layer][y], wires=y)

        return qml.probs(wires=list(range(n_qubits)))
    
    return _circuit(noise, class_angles, weights, n_qubits, q_depth)


def partial_measure(noise, class_angles, weights, n_qubits, q_depth, n_a_qubits):
    """
    Partial measurement - traces out ancillary qubits.
    Returns normalized probabilities WITHOUT forcing max to 1 (preserves variance).
    """
    probs = quantum_circuit(noise, class_angles, weights, n_qubits, q_depth)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    # Normalize to sum to 1, but DON'T divide by max (preserves variance)
    probsgiven0 = probsgiven0 / (torch.sum(probsgiven0) + 1e-8)
    return probsgiven0
