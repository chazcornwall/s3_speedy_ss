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


class BaselineTest(ITest):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        super().__init__(H, z, NT, initial_subset)

    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:
        solk = self._getSolution(H, z, subset)
        # Ignore bias term in SS statistic!
        stat = NT.test(solk[0][0:3,:], solr[0][0:3,:], solk[1][0:3,0:3], solr[1][0:3,0:3], lmbda=1e-5)
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

    