import numpy as np
from typing import Callable, List, Tuple, Dict, Union, Type
from scipy import stats
from pydantic import BaseModel, ConfigDict
import heapq
from copy import deepcopy
from abc import ABC, abstractmethod
import time


# Types
Subset = np.ndarray[Tuple[int], np.dtype[np.bool_]]
Stat = Tuple[bool, float]
Solution = Tuple[np.ndarray, np.ndarray, np.ndarray]
FdeOutput = Union[Tuple[Subset, Solution, float], float]


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


def chi2stat(H: np.ndarray, z: np.ndarray, sk: Subset) -> float:
    # Assume H and z have been normalized!
    Bk = create_B_from_subset(sk)
    Hk = Bk.T @ H
    zk = Bk.T @ z
    FI = Hk.T @ Hk # Fisher Information
    x = np.linalg.solve(FI, Hk.T @ zk)
    r = zk - Hk @ x
    return (r.T @ r / (zk.shape[0] - x.shape[0]))[0,0]


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


def ourheuristic(H: np.ndarray, z: np.ndarray, sr: Subset, sk: Subset) -> float:
    xr,Pr,_ = mle(H, z, sr)
    # Assume H and z have been normalized!
    Bk = create_B_from_subset(sk)
    Hk = Bk.T @ H
    zk = Bk.T @ z
    # print(f"our: {np.linalg.norm(Hk.T @ (zk - Hk @ xr))}")
    lb = np.sqrt(np.min(np.linalg.eigvals(Pr)).real) * np.linalg.norm(Hk.T @ (zk - Hk @ xr))
    return lb


def constheuristic(H: np.ndarray, z: np.ndarray, sr: Subset, sk: Subset) -> float:
    return 1.0


class NormalThreshold:

    def __init__(self, creq: float, dof: int = 1):
        """
        Calculate normalized SS statistic threshold (|x1 - x2|/sigma2) using continuity requirement, i.e. probability of false alarms:
        (1 - probthresh) = creq
        """
        assert dof > 0, "ChiSquare degrees of freedom (dof) must be an integer 1 or greater!"
        self.dof = dof
        self.T = stats.chi2.ppf(1 - creq, dof) # ppf is the inverse cdf

    def test(self, xk: np.ndarray, xr: np.ndarray, Pk: np.ndarray, Pr: np.ndarray, lmbda: float = 1.0) -> Stat:
        """
        Calculate SS statistic where xk has less measurements than xr
        """
        m = xk.shape[0]
        assert self.dof == m, "dof must equal the number of state dimensions!"
        xdelta = xr - xk
        Pdeltainv = np.linalg.inv(Pk - Pr + lmbda * np.eye(m))

        test_stat = float((xdelta.T @ Pdeltainv @ xdelta)[0,0])
        return (test_stat < self.T, test_stat)


class ITest(ABC):

    def __init__(self, H: np.ndarray, z: np.ndarray, NT: NormalThreshold, initial_subset: Subset):
        self.solution_hash_map : Dict[Tuple[bool, ...], Solution] = {}
        self._solr = self._getSolution(H, z, initial_subset)
        self._H = H
        self._z = z
        self._NT = NT

    def _getSolution(self, H: np.ndarray, z: np.ndarray, subset: Subset) -> Solution:
        try:
            sol = self.solution_hash_map[tuple(subset)]
        except KeyError:
            sol = mle(H, z, subset)
            self.solution_hash_map[tuple(subset)] = sol
        return sol

    def eval(self, subset: Subset) -> Stat:
        return self._eval(self._H, self._z, subset, self._NT, self._solr)

    def evalNoSubset(self) -> Union[bool, None]:
        return None

    @abstractmethod
    def _eval(self, H: np.ndarray, z: np.ndarray, subset: Subset, NT: NormalThreshold, solr: Solution) -> Stat:
        pass


class SSFDEParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    TestStatClass : Type[ITest]
    NT: NormalThreshold
    H: np.ndarray # Normalized measurement jacobian
    z: np.ndarray # Normalized measurement vector
    heuristic: str = "num"


class SSFDE:

    class PqType:
        counter = 0 # static class variable

        def __init__(self, subset: Subset = np.array([True]), meas: int = -1):
            SSFDE.PqType.counter += 1 
            self.counter = SSFDE.PqType.counter
            self.curr_subset : Subset = subset
            self.prev_meas : int = meas # This informs how to expand the current mask

        def __lt__(self, other):
            return self.counter < other.counter

    def __init__(self, params: SSFDEParams):
        self.__params = params
        self.__i = 0
        self.__duration_s = 10
        self.__duration_exceeded = False

        if self.__params.heuristic == "chi2":
            self.__heuristic = lambda H, z, init_subset, curr_subset: chi2stat(H, z, curr_subset) 
        elif self.__params.heuristic == "num":
            self.__heuristic = lambda H, z, init_subset, curr_subset: np.sum(curr_subset)
        elif self.__params.heuristic == "ours":
            self.__heuristic = lambda H, z, init_subset, curr_subset: ourheuristic(H, z, init_subset, curr_subset)
        elif self.__params.heuristic == "const":
            self.__heuristic = lambda H, z, init_subset, curr_subset: constheuristic(H, z, init_subset, curr_subset)
        else:
            self.__heuristic = lambda H, z, init_subset, curr_subset: np.sum(curr_subset)

    def fde(self, N: int, Nmin: int, max_duration_s: float = 10, verbose: bool = False) -> List[FdeOutput]:
        """
        Finds the largest, fault-free measurement subset.

        Params
        ---
        N : int, Number of measurements
        Nmin : int, Minimum number of measurements (Assumes maximum number of faults is N - Nmin)

        Returns
        ---
        Largest subset that passes the solution separation test or None if no fault-free subsets exist
        """
        # Initialize queue with all possible measurements
        assert N > Nmin, "N must be larger than Nmin!"
        pq : List[Tuple[float, SSFDE.PqType]] = []
        all_in_view_subset = np.array([True] * N)
        heapq.heappush(pq, (0, SSFDE.PqType(all_in_view_subset)))

        output : List[FdeOutput] = [] # Store all solutions up to the max duration

        # Add default solution
        best_sol : Solution = mle(self.__params.H, self.__params.z, all_in_view_subset)
        output.append((all_in_view_subset, best_sol, 0.0))

        self.__start_s = time.perf_counter()
        self.__duration_s = max_duration_s
        self.__duration_exceeded = False
        curr_best_subset = np.array([False] * N)

        # Try the all-in-view solution
        priority = self.eval_subset(all_in_view_subset, -1, Nmin, verbose=verbose)

        if priority:
            while pq:
                element : SSFDE.PqType = heapq.heappop(pq)[1]
                subset = element.curr_subset
                prev_meas = element.prev_meas

                # Expand each subset
                for i in range(prev_meas+1, N):
                    if subset[i]:
                        new_subset = deepcopy(subset)
                        new_subset[i] = False

                        # Ensure N > Nmin and better than current solution before performing SS test!
                        if max(Nmin, np.sum(curr_best_subset)) < np.sum(new_subset):
                            priority = self.eval_subset(new_subset, i, Nmin, verbose=verbose)

                            # If subset is not valid, try a smaller subset!
                            if priority:
                                heapq.heappush(pq, (priority, SSFDE.PqType(new_subset, i)))
                            else:
                                # Only keep the best subset (more measurements)!
                                if not self.__duration_exceeded:
                                    curr_best_subset = new_subset
                                    stop_s = time.perf_counter()
                                    output.append((curr_best_subset, mle(self.__params.H, self.__params.z, curr_best_subset), stop_s - self.__start_s))
                                    print(f"Found a better solution with {np.sum(curr_best_subset)} measurements!")
                                else:
                                    print(f"WARNING: Exceeded {self.__duration_s} s duration!")
                                    pq = [] # empty the queue! we are finished
                                    break
        else:
            curr_best_subset = all_in_view_subset

        stop_s = time.perf_counter()
        if np.sum(curr_best_subset) > Nmin: # Found solution that is not the trivial all_in_view solution
            best_sol : Solution = mle(self.__params.H, self.__params.z, curr_best_subset)
            output.append((curr_best_subset, best_sol, stop_s - self.__start_s))
        return output

    def eval_subset(self, subset: Subset, meas: int, Nmin: int, verbose: bool = True) -> Union[float, None]:
        """
        Prioritizes subsets that may contain a fault, e.g. test statistic passes but is very large. The faster we detect a fault,
        the faster we can stop iterating and move on to the next candidate.

        Returns
        ---
        A measure of how large the fault-free subset is or None if the subset contains a fault
        """
        if verbose:
            print(f"Evaluating subset: {subset}")

        test_stat : ITest = self.__params.TestStatClass(self.__params.H, self.__params.z, self.__params.NT, subset)

        # Initialize queue with a subset
        pq : List[Tuple[float, int, SSFDE.PqType]] = []
        heapq.heappush(pq, (0, SSFDE.PqType(subset)))

        # Try to make a decision before going through all the options using bounds
        quick_decision = test_stat.evalNoSubset() 
        if quick_decision is not None:
            if quick_decision:
                return None
            else:
                return self.__heuristic(self.__params.H, self.__params.z, subset, subset)

        while pq:
            element : SSFDE.PqType = heapq.heappop(pq)[1]
            subset = element.curr_subset
            prev_meas = element.prev_meas

            # Check for duration here because this is the highest frequency loop
            self.__duration_exceeded = time.perf_counter() - self.__start_s > self.__duration_s
            if self.__duration_exceeded:
                return None

            # Expand each subset to consider all possible combinations
            for i in range(prev_meas+1, len(subset)):
                if subset[i]:
                    new_subset = deepcopy(subset)
                    new_subset[i] = False

                    if np.sum(new_subset) >= Nmin:
                        passes, priority = test_stat.eval(new_subset)

                        if verbose:
                            print(f"\tSS test with subset: {new_subset}")
                            print(f"\t\tStat: {priority:0.3f}, Passes: {passes}")

                        # If the test passes, keep going! If not, quit!
                        if passes:
                            heapq.heappush(pq, (1.0 / priority, SSFDE.PqType(new_subset, i)))
                        else:
                            return self.__heuristic(self.__params.H, self.__params.z, subset, new_subset)
        return None

                        

            



        
