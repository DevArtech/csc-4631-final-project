"""
File: quantum_generator.py

Purpose:
This file defines the strongly-conditioned quantum circuit used as the generator
core for a Conditional Quantum Generative Adversarial Network (QGAN). It implements
a parameterized multi-layer quantum circuit that injects class-conditioning
information at every depth level to prevent feature washout and reduce mode collapse.
The file also includes a partial measurement routine that traces out ancillary
qubits while preserving output probability variance for stable training.

Key Components:
- Strongly-conditioned quantum circuit with repeated class injection
- Ring-topology entanglement for improved expressibility
- Parameter-shift differentiation for hybrid quantum-classical training
- Partial measurement for marginal probability extraction

Intended Use:
This module is used by the Conditional QGAN training pipeline to generate
class-conditioned probability distributions that are later mapped to image space.
"""

# Library imports
import torch
import pennylane as qml

def quantum_circuit(noise, class_angles, weights, n_qubits, q_depth):
    """
    Strongly-conditioned quantum circuit with class info injected at every layer.
    This prevents class information from being "washed out" and reduces mode collapse.
    
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

        # Initial encoding: noise + class conditioning
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
            qml.RX(class_rx[i], wires=i)
            qml.RZ(class_rz[i], wires=i)

        # Parameterized layers with entanglement AND repeated class conditioning
        for layer in range(q_depth):
            # Entanglement - use ring topology for better expressibility
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
            # Close the ring (last to first qubit)
            qml.CZ(wires=[n_qubits - 1, 0])
            
            # Parameterised rotations
            for y in range(n_qubits):
                qml.RY(weights[layer][y], wires=y)
            
            # Re-inject class conditioning at each layer (scaled by layer depth)
            # This prevents class info from being lost in deeper layers
            scale = 0.5 / (layer + 1)  # Diminishing but persistent class influence
            for y in range(n_qubits):
                qml.RZ(class_rz[y] * scale, wires=y)

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
