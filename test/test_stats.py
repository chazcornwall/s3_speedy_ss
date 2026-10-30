import ssfde
import utils
import numpy as np

def test_stats():

    NT = utils.NormalThreshold(1e-9)
    N = 10 # Measurement dimension
    M = 3 # State dimension

    # Create noise covariance matrix
    A = np.random.randn(N, N)
    R = A.T @ A
    L = np.linalg.cholesky(R)
    Linv = np.linalg.inv(L)

    # Generate a random measurement
    H = np.random.randn(N, M)
    x_gt = np.random.randn(M, 1)
    z = H @ x_gt + L @ np.random.randn(N,1)

    # Normalize measurement model
    Hnorm = Linv @ H
    znorm = Linv @ z

    s1 = utils.create_subset_mask([1], N)
    s2 = utils.create_subset_mask([1,2], N)

    print(utils.calc_ss_statistic_baseline(s1, s2, Hnorm, znorm, NT))
    print(utils.calc_ss_statistic_ours(s1, s2, Hnorm, znorm, NT))