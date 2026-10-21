import numpy as np
from typing import Callable, List, Tuple, Dict, Union
from pydantic import BaseModel
import heapq
from copy import deepcopy


Subset = List[bool]


class SSFDEParams(BaseModel):
    # A measure of how likely the subset will contain a fault or None if the subset does not pass the test
    eval_test_stat : Callable[[Subset, Subset], Union[float, None]]


class SSFDE:

    class PqType:

        def __init__(self, subset: Subset = [True], meas: int = 0):
            self.curr_subset : List[bool] = subset
            self.prev_meas : int = meas # This informs how to expand the current mask

    def __init__(self, params: SSFDEParams):
        self.__params = params

    def fde(self, N: int) -> Union[Subset, None]:
        """
        Finds the largest, fault-free measurement subset.

        Returns
        ---
        Largest subset that passes the solution separation test or None if no fault-free subsets exist
        """
        # Initialize queue with all possible measurements
        pq : List[Tuple[float, SSFDE.PqType]] = []
        heapq.heappush(pq, (0, [True] * N))

        while pq:
            element = heapq.heappop(pq)[1]
            subset = element.curr_subset
            prev_meas = element.prev_meas

            # Expand each subset
            for i in range(prev_meas+1, N):
                if subset[i]:
                    new_subset = deepcopy(subset)
                    new_subset[i] = False
                    priority = self.eval_subset(new_subset, i)

                    # If the subset is good, keep going! If not, do not try to evaluate it further!
                    if priority:
                        # Since we are taking away measurements, the first subset that passes will be the largest!
                        return new_subset
                    else:
                        heapq.heappush(pq, (priority, SSFDE.PqType(new_subset, i)))
        return None

    def eval_subset(self, subset: Subset, meas: int) -> Union[float, None]:
        """
        Prioritizes subsets that may contain a fault, e.g. test statistic passes but is very large. The faster we detect a fault,
        the faster we can stop iterating and move on to the next candidate.

        Returns
        ---
        A measure of how large the fault-free subset is or None if the subset contains a fault
        """

        # Initialize queue with a subset
        pq : List[Tuple[float, SSFDE.PqType]] = []
        heapq.heappush(pq, (meas, subset))

        while pq:
            element = heapq.heappop(pq)[1]
            subset = element.curr_subset
            prev_meas = element.prev_meas

            # Expand each subset
            for i in range(prev_meas+1, len(subset)):
                if subset[i]:
                    new_subset = deepcopy(subset)
                    new_subset[i] = False
                    priority = self.__params.eval_test_stat(subset, new_subset)

                    # If the test passes, keep going! If not, quit!
                    if priority:
                        heapq.heappush(pq, (priority, SSFDE.PqType(new_subset, i)))
                    else:
                        return None
        return sum(subset)

                        

            



        
