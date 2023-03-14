import numpy as np
import torch

# Pauli Matrices
def sigma_x():
    return torch.Tensor([[0,1],[1,0]])

def sigma_y():
    return torch.Tensor([[0,1j],[1j,0]])

def sigma_z():
    return torch.Tensor([[1,0],[0,-1]])

def i2():
    return torch.eye(2)

def operator_at(operator, index, N):
    """
    Description: Returns the tensor product of 2x2 identity matrices and
    the given operator at the appropriate index
    Parameters:
        operator (Tensor): operator acting on site [index]
        index (int): index of site
        N (int): number of sites
    Returns:
        op (Tensor): tensor product with operator acting on site
    """
    op = torch.eye(1)
    for i in range(N):
        if i == index:
            op = torch.kron(op, operator)
        else:
            op = torch.kron(op, i2())
    return op

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

def tfi_e0(N, J, h):
    # if N%2==0:
    #     m = np.arange(-(N-1)/2,(N+1)/2,1)
    # else:
    #     m = np.arange(-N/2,(N+1)/2,1/2)
    m = np.arange(-(N-1)/2,(N+1)/2,1)
    q = 2*np.pi*m/N
    lam = J/h
    omq = np.sqrt(1+2*lam*np.cos(q)+lam**2)
    return -np.sum(omq)*np.max([1/lam,1])
