import quantum_rbm_research.utils as utils

import numpy as np
import networkx as nx
from scipy.linalg import expm
import torch


class TransverseIsingHamiltonian():
    """
    Transverse-Field Ising Hamiltonian model with coupling J and transverse
    field h

    Currently only supports 1D Transverse Ising with N spins

    Select boundary conditions using obc
    """

    def __init__(self, N, J, h, obc=False):
        # Params
        self.N = N
        self.J = J
        self.h = h
        self.obc = obc

        # Create matrix
        self.H = self._H0() + self._H1()

    def e0_eig(self):
        """
        Description: Get ground state energy of self.H using minimum eigenvalue
        Returns:
            e0 (Tensor): ground state energy
        """
        e0 = torch.min(torch.real(torch.linalg.eigvals(self.H)))
        return e0

    def e0_analytic(self):
        """
        Description: calculate (analytically) the ground state energy for Model
        (from Quantum Ising Phase Transitions in Transverse Ising Models by
        Chakrabarti), with my own edits
        Returns:
            e0 (Tensor): ground state energy
        """
        m = torch.arange(-(self.N - 1) / 2, (self.N + 1) / 2, 1)
        q = 2 * torch.pi * m / self.N

        # lambda bar (pg. 13)
        lam = self.J / self.h

        # w_q (pg. 20)
        omega_q = torch.sqrt(1 + 2 * lam * torch.cos(q) + lam**2)

        # I noticed that for h > J, the returned value was off by a factor of
        # h from the minimum eigenvalue, which is why I added the second term.
        # (pg. 20)
        e0 = -torch.sum(omega_q) * max([self.h, 1])
        return e0

    def gs_eig(self):
        """
        Description: Get ground state vector of self.H using minimum
        eigenvector from numpy
        Returns:
            psi0 (Tensor): ground state vector
        """
        min_ind = torch.nonzero(
            torch.real(torch.linalg.eig(self.H).eigenvalues) == self.e0_eig()
        )

        psi0 = torch.flatten(
            torch.linalg.eig(self.H).eigenvectors[:, min_ind]
        )

        return psi0

    def gs_suzuki_trotter(self, tau, n, initial_state=None):
        """
        Description: Get ground state vector of self.H using Suzuki
        Trotter decompoisition and High Energy Filtering through Imaginary
        Time Propagation
        Parameters:
            tau (float): Time propagation
            n (int): num. Trotter steps
        Returns:
            psi0 (Tensor): ground state vector (should match eigenvector
            associated with min eigenvalue)
        """
        if initial_state is not None:
            psi0 = initial_state
        else:
            psi0 = torch.rand(2**self.N)
            psi0 /= torch.linalg.norm(psi0)
        delta_tau = tau / n

        # Suzuki Trotter decomp of imaginary time propagation
        time_prop = (
            torch.Tensor(expm(-delta_tau * self._H0()))
            @ torch.Tensor(expm(-delta_tau * self._H1()))
        )

        # time_prop2 = torch.linalg.matrix_power(time_prop, 4)

        # Operator n times
        for i in range(n):
            psi0 = time_prop @ psi0
            psi0 /= torch.linalg.norm(psi0)

        return psi0

    def convert_to_classical(self, tau, n):
        """
        Description: convert a TFI model to a classical spin model according
        to mapping on slide 16
        Parameters:
            tau (float): Time propagation parameter
            n (int): num of new dimensions
        Returns:
            isingHamiltonian (IsingHamiltonian): Ising Hamiltonian object
            representing classical conversion of self
        """
        # Suzuki trotter param
        delta_tau = tau / n

        N = self.N + (not self.obc)

        """
        Create graph using networkX
        Nxn lattice with weights between vertical, horizontal neighbors

        (N columns, n rows)
        NOTE: graph labels are [x,y], or [col, row]
        o-o-o-...-o
        | | | ... |
        o-o-o-...-o
            ...
        o-o-o-...-o
        """
        G = nx.grid_2d_graph(N, n)

        # Give all nodes a 'spin' attribute: +/- 1
        # Default to 1 for now
        for node in G.nodes:
            G.nodes[node]['spin'] = 1

        # Create IsingHamiltonian object with params
        isingHamiltonian = IsingHamiltonian(
            G,
            self.J,
            self.h,
            delta_tau,
            N,
            n)

        return isingHamiltonian

    def _H0(self):
        """
        Description: Return Hamiltonian matrix containing just interaction
        terms (J)
        Returns:
            H0 (Tensor): 2^N x 2^N matrix given by sum of adjacent spin-z
            operators
        """
        H0 = torch.zeros((2**self.N, 2**self.N))
        N = (self.N - 1) if self.obc else self.N
        for i in range(0, N):
            # Interaction term
            j = (i + 1) % self.N
            spinz_i = utils.operator_at(utils.sigma_z(), i, self.N)
            spinz_j = utils.operator_at(utils.sigma_z(), j, self.N)
            H0 += spinz_i @ spinz_j

        return -self.J * H0

    def _H1(self):
        """
        Description: Return Hamiltonian matrix containing just external field
        terms (h)
        Returns:
            H1 (Tensor): 2^N x 2^N matrix given by sum of adjacent spin-z
            operators
        """
        H1 = torch.zeros((2**self.N, 2**self.N))
        for i in range(0, self.N):
            # External field term
            spinx_i = utils.operator_at(utils.sigma_x(), i, self.N)
            H1 += spinx_i

        return -self.h * H1


class IsingHamiltonian():
    """
    Classical Ising Hamiltonian model. Used to transition from TFI to
    RBM. Particles and interactions are stored using networkx 2d graph.

    Spins are stored in node['spin'] = {+1, -1}

    Size of model depends on TFI and Trotter number (expansion in new
    dimension)
    """

    def __init__(self, graph, J, h, delta_tau, N, n):
        self.graph = graph
        self.J = J
        self.h = h
        self.delta_tau = delta_tau

    def update_weights(self):
        # Iterate through edges
        for edge in self.graph.edges:
            # if horizontal
            if abs(edge[0][0] - edge[1][0]) == 1:
                self.graph.edges[edge]['weight'] = (
                    self.delta_tau
                    * self.J
                    * self.graph.nodes[edge[0]]['spin']
                    * self.graph.nodes[edge[1]]['spin']
                )
            # else, vertical
            else:
                self.graph.edges[edge]['weight'] = (
                    -0.5
                    * np.log(np.tanh(self.delta_tau * self.h))
                    * self.graph.nodes[edge[0]]['spin']
                    * self.graph.nodes[edge[1]]['spin']
                )
            print(self.graph.get_edge_data(edge[0], edge[1]))
