import numpy as np
import torch

# Pauli Matrices
def sigma_x():
    return np.array([[0,1],[1,0]])

def sigma_y():
    return np.array([[0,1j],[1j,0]])

def sigma_z():
    return np.array([[1,0],[0,-1]])

def combinations(N):
    """
    Description: get all possible binary combinations for N bits
    Parameters:
        N (int): number of bits
    Returns:
        binary (Tensor): 2^N x N tensor, each row is a combination
    """
    binary = torch.Tensor([list(map(int,format(i, f'0{N}b'))) for i in range(2**N)])
    return binary
