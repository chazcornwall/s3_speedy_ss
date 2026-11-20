import numpy as np
from typing import List, Tuple, Dict, Callable

from truth.i_truth import ITruth, TruthType
from measurements.i_measurement import IMeasurement, MeasType
from ssfde import SSFDE, SSFDEParams


def create_measurement_model(x0: np.ndarray, bias: float, measurements: List[MeasType]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a measurement model (H) from a nominal position and available measurements
    """
    m = 4 # number of states (x, y, z, bias)
    n = len(measurements)
    H = np.zeros((n, m))
    z = np.zeros((n, 1))

    # R = LL^T where R is the measurement covariance
    L = np.diag(np.ones(n)*14.4) # https://github.com/cntaylor/FGO_gnss_integrity/blob/master/visualizations/RAIM_and_RCF_comparison.pdf
    Linv = np.linalg.inv(L)

    hx = np.zeros((n,1)) # Predicted pseudoranges
    for i in range(n):
        z[i, 0] = measurements[i][0]
        s = measurements[i][1]
        H[i,0:3] = (x0 - s).T / np.linalg.norm(x0 - s)
        H[i,3] = 1 # derivative of bias term
        hx[i, 0] = np.linalg.norm(x0 - s) + bias
    return Linv @ z, Linv @ H, Linv @ hx

# Use datasets from this directory! https://github.com/TUC-ProAut/libRSF/tree/master/datasets/UrbanNav

def evaluate(truth: ITruth, meas: IMeasurement, ssfdes: List[Callable[[np.ndarray, np.ndarray], SSFDE]], max_ssfde_duration_s: float = 0.1, time_range: Tuple[float, float] = (0, 100)):

    error : Dict[int, List] = {}
    for key in range(len(ssfdes)):
        error[key] = []

    # TODO: implement this error!
    anytime_error : Dict[int, List] = {}
    for key in range(len(ssfdes)):
        anytime_error[key] = []

    first = True
    x0_prev = np.zeros((3, 1))
    for tt in truth.getTimes():
        if time_range[0] <= tt and tt < time_range[1]:
            for mt in meas.getTimes():
                if time_range[0] <= mt and mt < time_range[1]:
                    if tt == mt:
                        if first:
                            x0_prev = truth.getTruth(tt) # Calculate nominal position
                            bias = 211300 # Estimated from data
                            first = False
                        else:
                            x0_curr = truth.getTruth(tt)
                            z, H, hx = create_measurement_model(x0_curr, bias, meas.getMeas(tt))
                            deltax_gt = x0_curr - x0_prev
                            x0_prev = x0_curr
                            for i, ssfde_cb in enumerate(ssfdes):
                                ssfde = ssfde_cb(H, z - hx) # Use z - h(x) as the "measurement" for an error state
                                outputs = ssfde.fde(z.shape[0], 4, max_duration_s=max_ssfde_duration_s)
                                output = outputs[-1] # Get last output (the best solution)
                                xhat, Phat, Ihat = output[1]
                                print(f"Excluded {z.shape[0] - np.sum(output[0])} measurements")
                                print(xhat)
                                print(deltax_gt)
                                error[i].append((xhat[0:3,:] - deltax_gt).T @ (xhat[0:3,:] - deltax_gt))

    # Print RMSE error after SSFDE (ATE of position): https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf
    for key in range(len(ssfdes)):
        sum = 0
        for err in error[key]:
            sum += err
        rmse = np.sqrt(sum / len(error[key]))
        print(f"RMSE for SSFDE {key}: {rmse}")
