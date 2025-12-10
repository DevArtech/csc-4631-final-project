"""
File: patch_quantum_generator.py

Purpose:
This file defines the PatchQuantumGenerator class, a strongly class-conditioned
hybrid quantum-classical generator used in a Conditional Quantum GAN (QGAN).
The generator employs multiple parameterized quantum sub-generators ("patches")
to produce quantum feature maps, which are then combined with deep class
embeddings and post-processed through a classical neural network to generate
final image outputs.

Key Features:
- Patch-based quantum generation using multiple quantum circuits
- Strong class conditioning with deep embeddings and residual connections
- Repeated class injection to prevent mode collapse
- LayerNorm-based classical post-processing for stability with small batch sizes
- Hybrid quantum-classical architecture for conditional image synthesis

Intended Use:
This module serves as the primary generator component in the Conditional QGAN
training pipeline, producing class-conditioned 8x8 grayscale digit images.
"""
#Imports
import math
import torch
import torch.nn as nn

from qcircuit import quantum_circuit, partial_measure

class PatchQuantumGenerator(nn.Module):
    """Conditional Quantum generator class with strong class conditioning to prevent mode collapse"""

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
        self.quantum_output_size = n_generators * self.patch_size

        # Quantum circuit parameters for each sub-generator
        # Initialize with wider spread for more diverse starting point
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits) * 2 - q_delta, requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        
        # Deeper class embedding with skip connection for stronger conditioning
        # This helps prevent the generator from ignoring class information
        self.class_embed_hidden = 64
        self.class_embedding_1 = nn.Sequential(
            nn.Linear(num_classes, self.class_embed_hidden),
            nn.LeakyReLU(0.2),
        )
        self.class_embedding_2 = nn.Sequential(
            nn.Linear(self.class_embed_hidden, self.class_embed_hidden),
            nn.LeakyReLU(0.2),
        )
        self.class_to_angles = nn.Sequential(
            nn.Linear(self.class_embed_hidden, n_qubits * 2),
            nn.Tanh()  # Output in [-1, 1], will scale to angles
        )
        
        # Separate class embedding for post-processing (prevents mode collapse)
        self.class_embed_post = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        
        # Classical post-processing WITHOUT BatchNorm (can cause mode collapse with small batches)
        # Uses LayerNorm instead for stable training without batch dependencies
        self.post_process = nn.Sequential(
            nn.Linear(self.quantum_output_size + 32, 128),  # 32 from class embedding
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
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
        
        # Deep class embedding with residual connection for stronger conditioning
        class_hidden = self.class_embedding_1(labels)
        class_hidden = class_hidden + self.class_embedding_2(class_hidden)  # Skip connection
        class_angles = self.class_to_angles(class_hidden) * math.pi  # Scale to [-pi, pi]

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
        
        # Separate class embedding for post-processing (stronger conditioning)
        class_post_features = self.class_embed_post(labels)
        
        # Concatenate quantum features with processed class features
        combined = torch.cat([quantum_features, class_post_features], dim=1)
        
        # Post-process to generate final image
        images = self.post_process(combined)

        return images
