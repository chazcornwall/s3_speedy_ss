import numpy as np
from scipy import stats
from ssfde import Subset, Stat, create_B_from_subset, Solution, ITest, NormalThreshold, mle, chi2stat, FdeOutput
from pydantic import BaseModel, ConfigDict
from typing import Dict, Tuple, Union, List, Callable
import time


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

class RandomMeasurementModel:

    def __init__(self, N: int, M: int, Nf: Union[int, Tuple], bias: Tuple):
        assert N >= M, "Must have more measurements than states!"
        # Create noise covariance matrix
        self.N = N
        self.M = M
        A = np.random.randn(N, N)
        self.R = A.T @ A
        self.L = np.linalg.cholesky(self.R)
        self.Linv = np.linalg.inv(self.L)

        # Generate a random measurement
        self.H = np.random.randn(N, M)
        self.Hnorm = self.Linv @ self.H
        self.x_gt = np.random.randn(M, 1)
        self.f, self.f_indices = fault_generator(self.N, Nf, self.M, bias)

    def sample(self):
        # Create random faults
        z = self.H @ self.x_gt + self.L @ np.random.randn(self.N,1) + self.L @ self.f

        # Normalize measurement model
        znorm = self.Linv @ z
        return self.Hnorm, znorm, self.x_gt, self.f_indices

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
        solk = self._getSolution(H, z, subset)
        stat = NT.test(solk[0], solr[0], solk[1], solr[1], lmbda=1e-8)
        return stat

    def evalNoSubset(self) -> bool | None:
        return super().evalNoSubset()


class BlanchTest(ITest):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        super().__init__(H, z, NT, initial_subset)
        self.__yr = z - H @ self._solr[0]
        self.__ub = (self.__yr.T * initial_subset) @ (self.__yr.T * initial_subset).T
        self.__ub = self.__yr.T @ self.__yr

    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:
        solk = self._getSolution(H, z, subset)
        stat = NT.test(solk[0], solr[0], solk[1], solr[1], lmbda=1e-8)
        return stat

    def evalNoSubset(self) -> bool | None:
        if self.__ub < self._NT.T:
            return True
        else:
            return None

    
class OurTest(ITest):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        super().__init__(H, z, NT, initial_subset)
        self.__yr = z - H @ self._solr[0]
        self.__ub = (self.__yr.T * initial_subset) @ (self.__yr.T * initial_subset).T
        self.__minPceig = np.min(np.linalg.eigvals(self._solr[1])).real

    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:
        dp = ((H.T * subset) @ (self.__yr.T * subset).T) # Hk.T @ Bk.T @ (self.__yr)
        lb = (dp.T @ dp * self.__minPceig)[0,0]

        if lb > NT.T:
            return (False, lb)
        
        solk = self._getSolution(H, z, subset)
        stat = NT.test(solk[0], solr[0], solk[1], solr[1], lmbda=1e-8)
        return stat

    def evalNoSubset(self) -> bool | None:
        if self.__ub < self._NT.T:
            return True
        else:
            return None


def create_subset_mask(faults: List[int], N: int) -> Subset:
    all_in_view_subset = np.array(N * [True], dtype=bool)
    for fault in faults:
        all_in_view_subset[fault] = False
    return all_in_view_subset


def get_output(query_elapsed_time_s: float, output: List[FdeOutput]) -> FdeOutput:
    curr_output = None
    for i in range(len(output)):
        if i == 0:
            assert query_elapsed_time_s >= output[i][2], f"Query time {query_elapsed_time_s} occurs before any data!"
        
        if i < len(output) - 1:
            if output[i][2] <= query_elapsed_time_s and query_elapsed_time_s < output[i+1][2]:
                curr_output = output[i]
                break
        else:
            curr_output = output[i]
            break
    return curr_output

    