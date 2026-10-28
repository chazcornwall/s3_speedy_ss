import numpy as np
import scipy as scp
from ssfde import Subset
from typing import Dict, Tuple

SOLUTION_HASH_MAP : Dict[Tuple[bool, ...], Tuple[np.ndarray, np.ndarray]] = {}

def calc_ss_statistic_baseline(s1: Subset, s2: Subset, H: np.ndarray, z: np.ndarray, xstat: bool = True, ystat: bool = False, zstat: bool = False):

    # Get solution for subset 1
    try:
        x1, P1 = SOLUTION_HASH_MAP[tuple(s1)]
    except KeyError:
        H1 = H[s1]
        FI1 = H1.T @ H1 # Fisher Information for H1 model
        x1 = np.linalg.solve(FI1, H1.T @ z)
        P1 = np.linalg.inv(FI1) 
        SOLUTION_HASH_MAP[tuple(s1)] = (x1, P1)

    # Get solution for subset 2
    try:
        x2, P2 = SOLUTION_HASH_MAP[tuple(s2)]
    except KeyError:
        H2 = H[s2]
        FI2 = H2.T @ H2 # Fisher Information for H2 model
        x2 = np.linalg.solve(FI2, H2.T @ z)
        P2 = np.linalg.inv(FI2) 
        SOLUTION_HASH_MAP[tuple(s2)] = (x2, P2)

    # Calculate statistics
    xdelta = x1 - x2
    Pdelta = P1 - P2 
    np.linalg.norm()