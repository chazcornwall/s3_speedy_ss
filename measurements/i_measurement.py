import numpy as np
import pickle
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

MeasType = Tuple[float, np.ndarray] # pseudorange and satellite location

class IMeasurement(ABC):

    def __init__(self):
        self._dict : Dict[float, List[MeasType]] = self._create_dict()

    @abstractmethod
    def _create_dict(self) -> Dict[float, List[MeasType]]:
        pass

    def getTimes(self) -> List[float]:
        return list(self._dict.keys())

    def getMeas(self, time: float) -> List[MeasType]:
        return self._dict[time]

    def save(self, filename: str):
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self._dict, file)

    def load(self, filename: str):
        with open(filename + ".pkl", "rb") as file:
            self._dict = pickle.load(file)
