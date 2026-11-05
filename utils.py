import numpy as np
from scipy import stats
from ssfde import Subset
from typing import Dict, Tuple, Union, List

Solution = Tuple[np.ndarray, np.ndarray]

Stat = Union[float, None]
SOLUTION_HASH_MAP : Dict[Tuple[bool, ...], Solution] = {}


# We assume only selecting for the vertical component (y) in the SS test statistics
Y = 1


class NormalThreshold:

    def __init__(self, creq: float):
        """
        Calculate normalized SS statistic threshold (|x1 - x2|/sigma2) using continuity requirement, i.e. probability of false alarms:
        (1 - probthresh) = creq
        """
        self.T = stats.chi2.ppf(1 - creq, 1) # ppf is the inverse cdf

    def test(self, x1: np.ndarray, x2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> Stat:
        # Calculate statistics
        xdelta = x1[Y, 0] - x2[Y, 0]
        sigma2delta = P1[Y, Y] - P2[Y, Y] 

        test_stat = xdelta**2 / sigma2delta
        if test_stat < self.T:
            return test_stat
        else:
            return None


def mle(H: np.ndarray, z: np.ndarray) -> Solution:
    """
    Calculate MLE estimate with measurement model z = Hx + w where w is unit-normal, white noise
    """
    FI = H.T @ H # Fisher Information for H2 model
    x = np.linalg.solve(FI, H.T @ z)
    P = np.linalg.inv(FI)
    return x, P 


def calc_ss_statistic_baseline(s1: Subset, s2: Subset, H: np.ndarray, z: np.ndarray, NT: NormalThreshold) -> Stat:

    # Get solution for subset 1
    try:
        x1, P1 = SOLUTION_HASH_MAP[tuple(s1)]
    except KeyError:
        x1, P1 = mle(H[s1], z[s1])
        SOLUTION_HASH_MAP[tuple(s1)] = (x1, P1)

    # Get solution for subset 2
    try:
        x2, P2 = SOLUTION_HASH_MAP[tuple(s2)]
    except KeyError:
        x2, P2 = mle(H[s2], z[s2])
        SOLUTION_HASH_MAP[tuple(s2)] = (x2, P2)

    return NT.test(x1, x2, P1, P2)


def calc_ss_statistic_ours(s1: Subset, s2: Subset, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, alpha: float = 1e-8) -> Stat:
    
    # Get solution for subset 1
    try:
        x1, P1 = SOLUTION_HASH_MAP[tuple(s1)]
    except KeyError:
        x1, P1 = mle(H[s1], z[s1])
        SOLUTION_HASH_MAP[tuple(s1)] = (x1, P1)

    H2 = H[s2]
    z2 = z[s2]
    d = np.array([[alpha, 1.0, alpha]]).T # Diagonal weighting matrix
    p = (H2.T * d) @ (z2 - H2 @ x1) # DH^T(z - Hx)

    # Calculate eigenvalue upper and lower bounds using Gershgorin circles 
    F = H2.T @ H2 # Fisher Information
    R = np.sum(np.abs(F), axis=1) - np.diag(F)
    eval_upper = np.max(np.diag(F) + R)
    eval_lower = np.min(np.diag(F) - R)

    print(f"F: {F}")
    print(f"eigval upper: {eval_upper}")
    print(f"eigval lower: {eval_lower}")

    sigma2_ub = 1 / eval_lower - P1[Y, Y]
    sigma2_lb = 1 / eval_upper - P1[Y, Y]

    if 1 / eval_upper < 1 / eval_lower:
        ss_ub = p[1,0] / (sigma2_lb * eval_lower)
        ss_lb = p[1,0] / (sigma2_ub * eval_upper)

        # If we can, early return using bounds!
        if ss_ub < NT.T:
            return ss_ub
        if NT.T < ss_lb:
            return None

    # Get solution for subset 2
    try:
        x2, P2 = SOLUTION_HASH_MAP[tuple(s2)]
    except KeyError:
        x2, P2 = mle(H[s2], z[s2])
        SOLUTION_HASH_MAP[tuple(s2)] = (x2, P2)

    return NT.test(x1, x2, P1, P2)


def create_subset_mask(faults: List[int], N: int) -> List[bool]:
    all_in_view_subset = N * [True]
    for fault in faults:
        all_in_view_subset[fault] = False
    return all_in_view_subset