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


def tfi_e0(N, J, h):
    """
    Description: calculate (analytically) the ground state energy for the given
    N, J, h (from Quantum Ising Phase Transitions in Transverse Ising Models
    by Chakrabarti, with my own edits)
    Parameters:
        N (int): number of spins
        J (int): interaction term (does this have to be an int?)
        h (int): transverse field term (does this have to be an int?)
    Returns:
        e0 (float): ground state energy
    """

    # m an q, but with my own changes (pg. 19)
    m = np.arange(-(N - 1) / 2, (N + 1) / 2, 1)
    q = 2 * np.pi * m / N

    # lambda bar (pg. 13)
    lam = J / h

    # w_q (pg. 20)
    omega_q = np.sqrt(1 + 2 * lam * np.cos(q) + lam ** 2)

    # I noticed that for h > J, the returned value was off by a factor of
    # J / h from the minimum eigenvalue, which is why I added the second term.
    # THIS MAY BE WRONG IF MY HAMILTONIAN MATRIX IS WRONG (MIN. EIGENVAL WRONG)
    return -np.sum(omega_q) * np.max([1 / lam, 1])

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
    """
    vis_col_int = np.arange(0, 2 ** (num_vis + num_hid)) // 2 ** num_hid
    hid_col_int = np.tile(np.arange(0, 2 ** num_hid), 2 ** num_vis)

    binary = pd.DataFrame({'vis': vis_col_int, 'hid': hid_col_int})
    binary['vis'] = binary['vis'].apply(
        lambda num: format(num, f'0{num_vis}b')
    )
    binary['hid'] = binary['hid'].apply(
        lambda num: format(num, f'0{num_hid}b')
    )

    return binary


def tensordist_to_dfdist(dist, num_vis, num_hid):
    """
    Description: convert a Tensor distribution into a DataFrame distribution
    Parameters:
        dist (Tensor): Tensor formatted as distribution
        num_vis (int): num vis nodes
        num_hid (int): num hid nodes
    Returns:
        dfdist (DataFrame): DataFrame, format of permutations_df with last
        column = probability, titled 'prob'
    """

    dfdist = permutations_df(num_vis, num_hid)
    dfdist['prob'] = dist[:, -1]
    return dfdist
