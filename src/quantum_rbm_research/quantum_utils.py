import numpy as np

# Pauli Matrices
def sigma_x():
    return np.array([[0,1],[1,0]])

def sigma_y():
    return np.array([[0,1j],[1j,0]])

def sigma_z():
    return np.array([[1,0],[0,-1]])

# Get all possible spin configurations for an N particle system
def spin_configs(N):
    # 2^N possible states
    binary = [bin(
