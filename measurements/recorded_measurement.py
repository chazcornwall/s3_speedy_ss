import numpy as np
from typing import Dict, Tuple, List


from measurements.i_measurement import IMeasurement, MeasType


class RecordedMeasurement(IMeasurement):

    def __init__(self, meas_file: str):
        self.__meas_file = meas_file
        super().__init__()

    def _create_dict(self) -> Dict[float, List[Tuple[float, np.ndarray]]]:
        dict : Dict[float, List[MeasType]] = {}
        with open(self.__meas_file, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 7 and parts[0] == 'pseudorange3':
                    try:
                        time_value = float(parts[1])
                        pseudorange_meas = float(parts[2])
                        sat_x = float(parts[4])
                        sat_y = float(parts[5])
                        sat_z = float(parts[6])
                        if time_value not in dict.keys():
                            dict[time_value] = []
                        dict[time_value].append((pseudorange_meas, np.array([[sat_x, sat_y, sat_z]]).T))
                    except:
                        print("Error parsing pseudorange line")
                        continue
        return dict

