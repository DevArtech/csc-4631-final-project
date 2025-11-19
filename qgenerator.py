import torch
import torch.nn as nn

from qcircuit import quantum_circuit, partial_measure

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, n_qubits, q_depth, n_a_qubits, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            n_qubits (int): Total number of qubits.
            q_depth (int): Depth of the parameterised quantum circuit.
            n_a_qubits (int): Number of ancillary qubits.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.n_generators = n_generators
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_a_qubits = n_a_qubits

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

    def forward(self, x):
        device = x.device
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)

        images = torch.Tensor(x.size(0), 0).to(device)

        for params in self.q_params:

            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params, self.n_qubits, self.q_depth, self.n_a_qubits).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            images = torch.cat((images, patches), 1)

        return images
