import math
import torch
import torch.nn as nn

from qcircuit import quantum_circuit, partial_measure

class PatchQuantumGenerator(nn.Module):
    """Conditional Quantum generator class with classical post-processing"""

    def __init__(self, n_generators, n_qubits, q_depth, n_a_qubits, num_classes=10, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            n_qubits (int): Total number of qubits.
            q_depth (int): Depth of the parameterised quantum circuit.
            n_a_qubits (int): Number of ancillary qubits.
            num_classes (int): Number of classes for conditional generation.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.n_generators = n_generators
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_a_qubits = n_a_qubits
        self.num_classes = num_classes
        
        # Calculate output size from quantum circuit
        self.patch_size = 2 ** (n_qubits - n_a_qubits)
        self.quantum_output_size = n_generators * self.patch_size  # 64 for default config

        # Quantum circuit parameters for each sub-generator
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        
        # Class embedding network: maps class labels to angles for quantum circuit
        # Output: n_qubits * 2 angles (for RX and RZ rotations)
        # Single shared embedding - balanced for stable training
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, n_qubits * 2),
            nn.Tanh()  # Output in [-1, 1], will scale to angles
        )
        
        # Classical post-processing: transforms quantum output + class info into final image
        # This gives the model more expressibility while keeping quantum core
        self.post_process = nn.Sequential(
            nn.Linear(self.quantum_output_size + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Tanh()  # Output in [-1, 1] to match image range
        )

    def forward(self, noise, labels):
        """
        Args:
            noise: Latent noise tensor of shape (batch_size, n_qubits)
            labels: One-hot encoded class labels of shape (batch_size, num_classes)
        
        Returns:
            Generated images of shape (batch_size, 64)
        """
        device = noise.device
        batch_size = noise.size(0)
        
        # Embed class labels to angles scaled to [-pi, pi] for rotation gates
        class_angles = self.class_embedding(labels) * math.pi  # (batch_size, n_qubits * 2)

        # Generate quantum features
        quantum_features = torch.Tensor(batch_size, 0).to(device)

        for params in self.q_params:
            patches = torch.Tensor(0, self.patch_size).to(device)
            
            for i in range(batch_size):
                q_out = partial_measure(
                    noise[i], 
                    class_angles[i], 
                    params, 
                    self.n_qubits, 
                    self.q_depth, 
                    self.n_a_qubits
                ).float().unsqueeze(0).to(device)
                patches = torch.cat((patches, q_out))

            quantum_features = torch.cat((quantum_features, patches), 1)
        
        # Concatenate quantum features with class labels for strong conditioning
        combined = torch.cat([quantum_features, labels], dim=1)
        
        # Post-process to generate final image
        images = self.post_process(combined)

        return images
