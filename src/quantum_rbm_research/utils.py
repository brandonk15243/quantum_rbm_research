import math
import numpy as np
import pandas as pd
import torch

########################################
# Quantum utils
########################################


# Pauli Matrices
def sigma_x():
    return torch.Tensor([[0, 1], [1, 0]])


def sigma_y():
    return torch.Tensor([[0, 1j], [1j, 0]])


def sigma_z():
    return torch.Tensor([[1, 0], [0, -1]])


# Identity
def i2():
    return torch.eye(2)


def operator_at(operator, index, N):
    """
    Description: Returns the tensor product of 2x2 identity matrices and the
    given operator at the appropriate index
    Parameters:
        operator (Tensor): operator acting at inedx
        index (int): index of site
        N (int): number of sites
    Returns:
        op (Tensor): 2^N matrix of tensor products
    """
    op = torch.eye(1)
    for i in range(N):
        if i == index:
            op = torch.kron(op, operator)
        else:
            op = torch.kron(op, i2())
    return op


def twospin_e0(J, h, tau, n, initial_state=None):
    """
    Description: calculate ground state vector using conversion to Classical
    lattice structure. Reference Appendix D in slides.
    Parameters:
        J (float): interaction factor
        h (float): external field factor
        tau (float): time propagation constant
        n (int): size of new dimension
        initial_state (Tensor): optional initial state
    Returns:
        gs (Tensor): ground state vector
    """

    # Arbitrary initial state
    if initial_state is not None:
        gs = initial_state
    else:
        gs = torch.rand(4)
    # Normalize
    gs /= np.linalg.norm(gs)

    # Suzuki Trotter params
    d_tau = tau / n

    # Time evolution matrix (1 step)
    matr = torch.zeros((4, 4))

    for i_l in range(4):
        # generate spin values sigma_1^l, sigma_2^l
        s1l = 1 if i_l < 2 else -1
        s2l = 1 if i_l % 2 == 0 else -1

        # generate basis vector |sigma_1^l sigma_2^l>
        ket = torch.zeros(4)
        ket[i_l] = 1.0

        for i_R in range(4):
            # generate spin values sigmna_1^R, sigma_2^R
            s1R = 1 if i_R < 2 else -1
            s2R = 1 if i_R % 2 == 0 else -1

            # generate basis vector <sigma_1^R sigmna_2^R|
            bra = torch.zeros(4)
            bra[i_R] = 1.0

            outer_prod = torch.outer(ket, bra)

            outer_prod *= (
                np.exp(d_tau * J * s1l * s2l)
                * np.exp(-0.5 * np.log(np.tanh(d_tau * h)) * s1l * s1R)
                * np.exp(-0.5 * np.log(np.tanh(d_tau * h)) * s2l * s2R)
            )

            matr += outer_prod

    # Apply operator n times
    for i in range(n):
        gs = matr @ gs
        gs /= np.linalg.norm(gs)

    return gs

########################################
# MISC utils
########################################


def permutations(N):
    """
    Description: get all possible permutations for N bits (start at 0)
    Parameters:
        N (int): number of bits
    Returns:
        binary (Tensor): 2^N x N tensor, each row is a permutation
    """
    binary = torch.Tensor(
        [list(
            map(
                int, format(i, f'0{N}b'))) for i in range(2**N)]
    )
    return binary


def permutations_df(num_vis, num_hid):
    """
    Description: get all possible permutations for N bits (start at 0)
    Parameters:
        N (int): number of bits
    Returns:
        binary (DataFrame): 2^N x 2 DF
            first column = vis config
            second column = hid config
            third column = full config
    """
    vis_col_int = np.arange(0, 2**(num_vis + num_hid)) // 2**num_hid
    hid_col_int = np.tile(np.arange(0, 2**num_hid), 2**num_vis)

    binary = pd.DataFrame({'vis': vis_col_int, 'hid': hid_col_int})
    binary['vis'] = binary['vis'].apply(
        lambda num: format(num, f'0{num_vis}b')
    )
    binary['hid'] = binary['hid'].apply(
        lambda num: format(num, f'0{num_hid}b')
    )
    binary['full'] = binary['vis'].str.cat(binary['hid'], sep=',')

    return binary


def dfdist_from_RBM(model):
    """
    Description: Generate dfdist (DataFrame) for given model
    Parameters:
        model (RBM): RBM to generate distribution from
    Returns:
        dfdist (DataFrame): dfdist
    """
    dfdist = permutations_df(model.num_vis, model.num_hid)
    dfdist['boltz'] = model.get_boltzmann_distribution()[:, -1]
    dfdist['gibbs'] = model.get_gibbs_distribution()[:, -1]

    return dfdist
