import quantum_rbm_research.utils as utils
import quantum_rbm_research.Models as Models

import numpy as np
import networkx as nx
from scipy.linalg import expm, eig, inv

import torch


class TransverseIsingHamiltonian():
    """
    Transverse-Field Ising Hamiltonian model

        H_Q = -(sum_i^M J_{i,i+1} * sigma_i^z * sigma_{i+1}^z
                + sum_i^M h_x * sigma_i^x
                + sum_i^M h_z * sigma_i^z

    Currently only supports 1D

    First summation (coupling summation) will change based on boundary
    conditions, set by obc

    Parameters:
        N (int):        number of spins
        J (float/arr):  coupling term. If scalar, same coupling for all
                        spins. If arr (should have size N), values for
                        spin between {i, i+1} is J[i]
        h_x (float):    transverse field strength
        h_z (float):    z-pointing field. Used for symmetry breaking.
                        Default 0
    """

    def __init__(self, N, J, h_x, h_z=0.0, obc=False):
        # Params
        self.N = N
        self.J = J
        self.h_x = h_x
        self.h_z = h_z
        self.obc = obc

        # Create matrix
        self.H = self._H0() + self._H1()

    def e0_eig(self):
        """
        Description: Ground state energy using eigenvalues
        Returns:
            e0 (Tensor):    ground state energy
        """
        e0 = torch.min(torch.real(torch.linalg.eigvals(self.H)))
        return e0

    def e0_analytic(self):
        """
        Description: analytical ground state energy. DOES NOT WORK WITH
        A NON-ZERO h_z!!!
        (from Quantum Ising Phase Transitions in Transverse Ising Models by
        Chakrabarti), with my own edits
        Returns:
            e0 (Tensor):    ground state energy
        """
        if self.h_z != 0:
            raise NotImplementedError

        m = torch.arange(-(self.N - 1) / 2, (self.N + 1) / 2, 1)
        q = 2 * torch.pi * m / self.N

        # lambda bar (pg. 13)
        lam = self.J / self.h_x

        # w_q (pg. 20)
        omega_q = torch.sqrt(1 + 2 * lam * torch.cos(q) + lam**2)

        # I noticed that for h > J, the returned value was off by a factor of
        # h from the minimum eigenvalue, which is why I added the second term.
        # (pg. 20)
        e0 = -torch.sum(omega_q) * max([self.h_x, 1])
        return e0

    def gs_eig(self):
        """
        Description: Ground state vector of self.H using eigenvectors
        Returns:
            psi0 (Tensor):  ground state vector
        """
        min_ind = torch.nonzero(
            torch.real(torch.linalg.eig(self.H).eigenvalues) == self.e0_eig()
        )

        psi0 = torch.flatten(
            torch.linalg.eig(self.H).eigenvectors[:, min_ind]
        )

        return psi0

    def gs_suzuki_trotter(self, tau, n, init_vec=None):
        """
        Description: Get ground state vector of self.H using Suzuki
        Trotter decompoisition and high energy filtering through imaginary
        time propagation
        Parameters:
            tau (float):        Time propagation
            n (int):            num. Trotter steps
            init_vec (Tensor):  initial state tensor
        Returns:
            psi0 (Tensor): ground state vector
        """
        if init_vec is not None:
            psi0 = init_vec
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

    def convert_self_to_classical(self, tau, n):
        """
        Description: Convert self to classical model using Suzuki Trotter
        Parameters:
            tau (float): Time propagation parameter
            n (int): num of new dimensions
        Returns:
            ising (IsingHamiltonian): Ising Hamiltonian object
            representing classical conversion of self
        """

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
        G = nx.grid_2d_graph(self.N, n)

        # Give all nodes a 'spin' attribute: +/- 1
        # Default to 1
        nx.set_node_attributes(G, 1, 'spin')

        # Create IsingHamiltonian object with params
        isingHamiltonian = IsingHamiltonian(
            self.N,
            self.J,
            self.h_x,
            tau,
            n,
            h_z=self.h_z,
            graph=G,
            obc=self.obc,
            parent=self)

        return isingHamiltonian

    def avg_z_analytic(self, beta=10.0):
        """
        Description: Average z spin
        Parameters:
            beta (float):   1/(boltz. const.) * (temp)
                         note: boltz_const = 1
        Returns:
            avg_z (float):  average z spin
        """

        # Construct diagonal matrix with gs energy
        e0_diag = np.zeros_like(self.H)
        np.fill_diagonal(e0_diag, self.e0_eig())

        # Subtract
        subtracted = self.H - e0_diag

        S_op = torch.zeros((2**self.N, 2**self.N))

        for i in range(self.N):
            S_op += utils.operator_at(utils.sigma_z(), i, self.N)

        S_op /= self.N

        num = np.trace(S_op @ expm(-beta * subtracted), dtype=float)
        den = np.trace(expm(-beta * subtracted), dtype=float)

        return num / den

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
            and spin-x operators
        """
        H1 = torch.zeros((2**self.N, 2**self.N))
        for i in range(0, self.N):
            # External field term
            spinx_i = -self.h_x * utils.operator_at(utils.sigma_x(), i, self.N)
            spinz_i = -self.h_z * utils.operator_at(utils.sigma_z(), i, self.N)
            H1 += spinx_i + spinz_i

        return H1


class IsingHamiltonian():
    """
    Classical Ising Hamiltonian model. Used to transition from TFI to
    RBM. Particles and interactions are stored using networkx 2d graph.

    Spins are stored in node['spin'] = {+1, -1}

    Size of model depends on TFI and Trotter number (expansion in new
    dimension)

    Z-DIRECTED FIELD NOT IMPLEMENTED YET!

    Parameters:
        N, J, h_x, h_z, obc:    (see Transverse Ising Hamiltonian)
        tau (float):            imaginary time step
        n (int):                Trotter number, size in new dimension
        graph (Graph):          networkX graph representing graph
        parent (TransverseIsingHamiltonain): pointer to parent
    """

    def __init__(self, N, J, h_x, tau, n, h_z=0.0, graph=None, obc=False,
                 parent=None):
        self.N = N
        self.J = J
        self.h_x = h_x
        self.h_z = h_z
        self.tau = tau
        self.n = n
        delta_tau = tau / n
        self.bias_z = delta_tau * self.h_z
        if graph is None:
            self.graph = nx.grid_2d_graph(N, n)
            nx.set_node_attributes(self.graph, 1, 'spin')
            nx.set_node_attributes(self.graph, self.bias_z, 'bias')
        else:
            self.graph = graph
        self.obc = obc
        self.parent = parent

        # WH is horizontal coupling,
        # WV is vertical coupling

        # In case of 0 h_x, set to small value
        if self.h_x == 0:
            self.h_x == 1e-3
        self.WH = delta_tau * self.J
        self.WV = -0.5 * np.log(np.tanh(delta_tau * self.h_x))

        # Iterate through edges
        for edge in self.graph.edges:
            # if horizontal
            if abs(edge[0][0] - edge[1][0]) == 1:
                self.graph.edges[edge]['weight'] = self.WH
            # else, vertical
            else:
                self.graph.edges[edge]['weight'] = self.WV

    def convert_self_to_rbm2d(self):
        # Hidden nodes are (2xnxN)
        hid = torch.zeros((2, self.n, self.N))

        # Visible nodes are (1xNxn)
        vis = torch.zeros((1, self.n, self.N))

        return Models.RBM2D(
            vis,
            hid,
            self.WH,
            self.WV,
            self.bias_z,
            self.N,
            self.n,
            obc=self.obc,
            parent=self
        )
