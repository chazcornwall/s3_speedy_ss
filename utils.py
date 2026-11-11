import numpy as np
from scipy import stats
from ssfde import Subset, Stat, create_B_from_subset, Solution, ITest, NormalThreshold, mle, chi2stat
from pydantic import BaseModel, ConfigDict
from typing import Dict, Tuple, Union, List


SOLUTION_HASH_MAP : Dict[Tuple[bool, ...], Solution] = {}


# We assume only selecting for the vertical component (y) in the SS test statistics
Y = 1


def bounded_subset(sf: Subset, sk: Subset, sc: Subset) -> bool:
    """
    Check if sk is bounded by sf (s "floor") and sc (s "ceiling")
    """
    return np.array_equal(sf, sf * sk) and np.array_equal(sk, sk * sc)

def sym_sqrt(A: np.ndarray) -> np.ndarray:
    eigvals, Q = np.linalg.eig(A)
    Lambda = np.diag(eigvals)
    return Q @ np.sqrt(Lambda) @ Q.T

def fault_generator(N: int, Nf: Union[int, Tuple], Nmin, bias: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(Nf, int):
        assert Nf < N, f"Number of faults {Nf} must be less than number of measurements {N}!"
        assert N - Nf >= Nmin, f"Number of fault-free measurements {N - Nf} must be greater than {Nmin}"
    else:
        Nf = np.random.randint(Nf[0], Nf[1]+1)
        assert Nf < N, f"Number of faults {Nf} must be less than number of measurements {N}!"
        assert N - Nf >= Nmin, f"Number of fault-free measurements {N - Nf} must be greater than {Nmin}"
    fault_indices = np.random.choice(N, Nf, replace=False)
    fault = np.zeros((N, 1))
    for i in fault_indices:
        fault[i, 0] = (bias[1] - bias[0]) * np.random.rand() + bias[0]
    print(fault)
    return fault, fault_indices

def random_measurement_model_with_faults(N: int, Nf: Union[int, Tuple], M: int, bias: Tuple):
    assert N >= M, "Must have more measurements than states!"
    # Create noise covariance matrix
    A = np.random.randn(N, N)
    R = A.T @ A
    L = np.linalg.cholesky(R)
    Linv = np.linalg.inv(L)

    # Create random faults
    f, f_indices = fault_generator(N, Nf, M, bias)

    # Generate a random measurement
    H = np.random.randn(N, M)
    x_gt = np.random.randn(M, 1)
    z = H @ x_gt + L @ np.random.randn(N,1) + L @ f

    # Normalize measurement model
    Hnorm = Linv @ H
    znorm = Linv @ z
    return Hnorm, znorm, x_gt, f_indices

class SolutionSeparationBoundsParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    lmbda: float = 1.0
    blanch: bool = False
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

        # Get upperbound for the spectral norm of Hc.T @ Hc
        # self.__ubIcnorm2 = np.max(np.linalg.eigvals(self.__params.Solc[2]))**2 # np.trace(self.__params.Solc[2])**2
        self.__ubIcnorm2 = np.min(np.linalg.eigvals(self.__params.Solc[1]))**2
        print(self.__ubIcnorm2)

        # Get upperbound for the spectral norm of Pf = (Hf.T @ Hf)^{-1}
        self.__ubPfnorm2 = np.max(np.linalg.eigvals(self.__params.Solf[1]))**2 #  np.trace(self.__params.Solf[1])**2

        self.__Pr = self.__params.Solr[1]
        m = self.__params.H.shape[1]

        # Calculate upperbound for the spectral norm of the square root of Pf - Pr + Lambda, which is PSD wrt Pk - Pr + Lambda
        self.__Pf = self.__params.Solf[1]
        Pdeltaf = self.__Pf - self.__Pr + self.__params.lmbda * np.eye(m)
        Pdeltaf_eigvals = np.linalg.eigvals(np.linalg.inv(Pdeltaf))
        self.__ubsqrtPfPrnorm2 = np.min(Pdeltaf_eigvals) # np.trace(Q @ np.sqrt(Lambda) @ Q.T)**2 # 

        # Calculate upperbound for the spectral norm of the square root of Pc - Pr + Lambda, which is negative semi-definite wrt Pk - Pr + Lambda
        self.__Pc = self.__params.Solc[1]
        Pdeltac = self.__Pc - self.__Pr + self.__params.lmbda * np.eye(m)
        invPdeltac_eigvals = np.linalg.eigvals(np.linalg.inv(Pdeltac))
        self.__ubinvsqrtPcPrnorm2 = np.max(invPdeltac_eigvals) # np.trace(Q @ np.sqrt(Lambda) @ Q.T)**2 #

        self.__xr = self.__params.Solr[0]
        self.__yr = self.__params.z - self.__params.H @ self.__xr

        print(np.max(np.linalg.eigvals(self.__Pf - self.__Pr)))

        # Blanch upperbounds the solution separation using the SS covariance. We do it slightly differently using an indirect
        # derivation of the SS statistic.
        self.__ub_blanch = np.linalg.norm(self.__yr)**2  # * np.max(np.linalg.eigvals(self.__Pf - self.__Pr)) * self.__ubinvsqrtPcPrnorm2
        self.__lb = self.__ubsqrtPfPrnorm2 * np.linalg.norm(self.__yr)**2 * np.min(np.linalg.eigvals(self.__Pc - self.__Pr))
        

    def calcBounds(self, sk: Subset, verbose: bool = False) -> Tuple[float, float]:
        """
        Calculates bounds for the statistic: (xk - xr)^2 / (sigma2k - sigma2r)
        """
        assert bounded_subset(self.__params.sf, sk, self.__params.sc), "Current subset (sk) is not bounded by Floor subset (sf) and Ceiling subset (sc)"

        dp = ((self.__params.H.T * sk) @ (self.__yr.T * sk).T) # Hk.T @ Bk.T @ (self.__yr)
        dpnorm2 = dp.T @ dp

        if verbose:
            print(f"dpnorm2 = {dpnorm2}")
            print(dpnorm2 * self.__ubPfnorm2)
            print(np.max(np.linalg.eigvals(self.__Pf - self.__Pr)) * np.linalg.norm(self.__yr)**2)
            print(f"||sqrt(Pf - Pr)||_2^2 = {self.__ubsqrtPfPrnorm2}")
            print(f"||sqrt(inv(Pc - Pr))||_2^2 = {self.__ubinvsqrtPcPrnorm2}")
            print(f"||Hc^T Hc||_2^2 = {self.__ubIcnorm2}")

        # lb = dpnorm2 * self.__ubsqrtPfPrnorm2 * self.__ubIcnorm2
        lb = dpnorm2 * self.__ubIcnorm2**(1/2)
        if self.__params.blanch:
            ub = self.__ub_blanch
        else:
            ub = dpnorm2 * self.__ubinvsqrtPcPrnorm2 * self.__ubPfnorm2
            # ub = dpnorm2 * self.__ubPfnorm2
        return lb, ub


class BaselineTest(ITest):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        super().__init__(H, z, NT, initial_subset)

    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:
        try:
            xk, Pk,_ = SOLUTION_HASH_MAP[tuple(subset)]
        except KeyError:
            xk, Pk, FIk = mle(H, z, subset)
            SOLUTION_HASH_MAP[tuple(subset)] = (xk, Pk, FIk)

        return NT.test(xk, solr[0], Pk, solr[1], lmbda=1e-8)

class BlanchTest(ITest):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        super().__init__(H, z, NT, initial_subset)
        self.__ub = chi2stat(H, z, initial_subset)

    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:

        if self.__ub < NT.T:
            return (True, self.__ub)
        
        try:
            xk, Pk,_ = SOLUTION_HASH_MAP[tuple(subset)]
        except KeyError:
            xk, Pk, FIk = mle(H, z, subset)
            SOLUTION_HASH_MAP[tuple(subset)] = (xk, Pk, FIk)

        return NT.test(xk, solr[0], Pk, solr[1], lmbda=1e-8)

class OurTest(ITest):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        super().__init__(H, z, NT, initial_subset)
        self.__ub = chi2stat(H, z, initial_subset)

        sol = mle(H, z, initial_subset)
        self.__minPceig = np.min(np.linalg.eigvals(sol[1]))
        self.__yr = z - H @ sol[0]

    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:

        if self.__ub < NT.T:
            return (True, self.__ub)

        dp = ((H.T * subset) @ (self.__yr.T * subset).T) # Hk.T @ Bk.T @ (self.__yr)
        dpnorm2 = dp.T @ dp
        lb = dpnorm2 * self.__minPceig

        if lb > NT.T:
            return (False, lb)
        
        try:
            xk, Pk,_ = SOLUTION_HASH_MAP[tuple(subset)]
        except KeyError:
            xk, Pk, FIk = mle(H, z, subset)
            SOLUTION_HASH_MAP[tuple(subset)] = (xk, Pk, FIk)

        return NT.test(xk, solr[0], Pk, solr[1], lmbda=1e-8)


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