import numpy as np
from typing import Dict, List

from truth.i_truth import ITruth, TruthType


class RecordedTruth(ITruth):

    def __init__(self, truth_file: str):
        super().__init__()
        """
        Store true position of ego object
        """
        self.__truth_file = truth_file

    def _create_dict(self) -> Dict[float, TruthType]:
        dict : Dict[float, np.ndarray] = {}
        with open(self.__truth_file, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == 'point3':
                    try:
                        time_value = float(parts[1])
                        x_loc = float(parts[2])
                        y_loc = float(parts[3])
                        z_loc = float(parts[4])
                        dict[time_value] = np.array([[x_loc, y_loc, z_loc]]).T
                    except:
                        print("Skipping line due to invalid number format: %s", line.strip())
                        continue
                else:
                    print("Invalid line in truth file: %s", line.strip())
        return dict