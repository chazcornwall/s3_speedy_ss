import numpy as np
import pickle
from pydantic import BaseClass
from typing import Dict, List
from abc import ABC, abstractmethod

from measurements.i_measurement import IMeasurement, MeasType
from measurements.recorded_measurement import RecordedMeasurement
from truth.recorded_truth import ITruth


class GeneratedMeasurementParams(BaseClass):
    stddev: float
    fault_prob: float
    fault_magnitude: float


class GeneratedMeasurement(IMeasurement):

    def __init__(self, truth: ITruth, recorded_measurement: RecordedMeasurement, params: GeneratedMeasurementParams):
        super().__init__()
        self.__params = params
        self.__truth = truth
        self.__rm = recorded_measurement

    def _create_dict(self) -> Dict[float, List[MeasType]]:
        dict : Dict[float, List[MeasType]] = {}
        truth_times = self.__truth.getTimes()
        recorded_measurement_times = self.__rm.getTimes()

        for tt in truth_times:
            for rmt in recorded_measurement_times:
                if tt == rmt:
                    dict[tt] = []
                    for meas in self.__rm.getMeas(rmt):
                        sat_pos = meas[1]
                        pseudorange_mean = np.linalg.norm(self.__truth.getTruth(tt) - sat_pos)

                        # Inject faults according to fault probability
                        if np.random.rand() < self.__params.fault_prob:
                            pseudorange_mean += self.__params.fault_magnitude
                        pseudorange = pseudorange_mean + self.__params.stddev * np.random.randn()

                        # Create new measurement
                        dict[tt].append((pseudorange, sat_pos))
        return dict


