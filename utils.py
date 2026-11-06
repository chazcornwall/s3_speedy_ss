import numpy as np
from scipy import stats
from ssfde import Subset
from pydantic import BaseModel, ConfigDict
from typing import Dict, Tuple, Union, List

Solution = Tuple[np.ndarray, np.ndarray, np.ndarray]

Stat = Tuple[bool, float]
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

    def test(self, xk: np.ndarray, xr: np.ndarray, Pk: np.ndarray, Pr: np.ndarray, index: int) -> Stat:
        """
        Calculate SS statistic where xk has less measurements than xr
        """
        assert index >= 0 and index < xk.shape[0], "Index must be a valid state index!"
        xdelta = xr[index, 0] - xk[index, 0]
        sigma2delta = Pk[index, index] - Pr[index, index] 

        test_stat = xdelta**2 / sigma2delta
        return (test_stat < self.T, test_stat)


def bounded_subset(sf: Subset, sk: Subset, sc: Subset) -> bool:
    """
    Check if sk is bounded by sf (s "floor") and sc (s "ceiling")
    """
    return np.array_equal(sf, sf * sk) and np.array_equal(sk, sk * sc)

def create_B_from_subset(s: Subset) -> np.ndarray:
    n_subset = int(np.sum(s)) # Number of measurements in the subset
    n = s.shape[0] # Total number of measurements
    B = np.zeros((n, n_subset))

    k = 0
    for i in range(n_subset):
        for j in range(k, n):
            if s[j]:
                B[j, i] = 1.0
                k = j + 1
                break
    return B


class SolutionSeparationBoundsParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    H: np.ndarray
    z: np.ndarray
    Solr: Solution
    Solc: Solution
    Solf: Solution
    sr: Subset
    sc: Subset
    sf: Subset


class SolutionSeparationBounds:

    def __init__(self, params: SolutionSeparationBoundsParams):
        self.__params = params
        assert bounded_subset(self.__params.sf, self.__params.sc, self.__params.sr), "Ceiling subset (sc) is not bounded by Floor subset (sf) and Roof subset (sr)!"
        
        self.__trIc = np.trace(self.__params.Solc[2])
        self.__Pc = self.__params.Solc[1]

        self.__Pf = self.__params.Solf[1]
        self.__trPf = np.trace(self.__params.Solf[1])

        self.__Pr = self.__params.Solr[1]

        Br = create_B_from_subset(self.__params.sr)
        self.__Hr = Br.T @ self.__params.H
        self.__xr = self.__params.Solr[0]
        self.__yr = self.__params.z - self.__params.H @ self.__xr
        

    def calcBounds(self, sk: Subset, index: int) -> Tuple[float, float]:
        """
        Calculates bounds for the statistic: (xk - xr)^2 / (sigma2k - sigma2r)
        """
        assert bounded_subset(self.__params.sf, sk, self.__params.sc), "Current subset (sk) is not bounded by Floor subset (sf) and Ceiling subset (sc)"
        assert index >= 0 and index < self.__xr.shape[0], "Index must be a valid state index!"

        dp = ((self.__params.H.T * sk)[[index], :] @ (self.__yr.T * sk).T)**2
        sigma2f = self.__Pf[index, index]
        sigma2r = self.__Pr[index, index]
        sigma2c = self.__Pc[index, index]

        lb = dp / (self.__trIc * (sigma2f - sigma2r))
        ub = self.__trPf * dp / (sigma2c - sigma2r) 
        return lb, ub


def mle(H: np.ndarray, z: np.ndarray, sk: Subset) -> Solution:
    """
    Calculate MLE estimate with measurement model z = Hx + w where w is unit-normal, white noise
    """
    Bk = create_B_from_subset(sk)
    Hk = Bk.T @ H
    zk = Bk.T @ z
    FI = Hk.T @ Hk # Fisher Information
    x = np.linalg.solve(FI, Hk.T @ zk)
    P = np.linalg.inv(FI)
    return x, P, FI


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


def create_subset_mask(faults: List[int], N: int) -> Subset:
    all_in_view_subset = np.array(N * [True], dtype=bool)
    for fault in faults:
        all_in_view_subset[fault] = False
    return all_in_view_subset