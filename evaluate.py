import numpy as np
from typing import List, Tuple, Dict

from truth.i_truth import ITruth, TruthType
from measurements.i_measurement import IMeasurement, MeasType
from ssfde import SSFDE, SSFDEParams


def create_measurement_model(x0: np.ndarray, measurements: List[MeasType]) -> Tuple[np.ndarray, np.ndarray]
    """
    Create a measurement model (H) from a nominal position and available measurements
    """
    m = 4 # number of states (x, y, z, bias)
    n = len(measurements)
    H = np.zeros((n, m))
    z = np.zeros((n, 1))

    for i in range(n):
        z[i, 0] = measurements[i][0]
        s = measurements[i][1]
        H[i,0:3] = (x0 - s).T / np.linalg.norm(x0 - s)
        H[i,3] = 1 # derivative of bias term
    return z, H

# Use datasets from this directory! https://github.com/TUC-ProAut/libRSF/tree/master/datasets/UrbanNav

def evaluate(truth: ITruth, meas: IMeasurement, ssfdes: List[SSFDE]):

    # Use RMSE error! (ATE of position): https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf
    error : Dict[int, List] = {}
    for key in range(len(ssfdes)):
        error[key] = []

    first = True
    x0 = np.zeros((3, 1))
    for tt in truth.getTimes():
        for mt in meas.getTimes():
            if tt == mt:
                if first:
                    x0 = truth.getTruth(tt) # Calculate nominal position
                    first = False
                else:
                    z, H = create_measurement_model(x0, meas.getMeas(tt))
                    for i, ssfde in enumerate(ssfdes):
                        outputs = ssfde.fde(z.shape[0], 4, max_duration_s=1.0)
                        output = outputs[-1] # Get last output (the best solution)
                        xhat, Phat, Ihat = output[1]
                        x0 = truth.getTruth(tt)
                        error[i].append((xhat - x0).T @ (xhat - x0))

    # Print RMSE error after SSFDE
    for key in range(len(ssfdes)):
        sum = 0
        for err in error[key]:
            sum += err
        rmse = np.sqrt(sum / len(error[key]))
        print(f"RMSE for SSFDE {key}: {rmse}")
