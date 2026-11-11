import ssfde
import utils
import numpy as np
from itertools import combinations
import time

def test_B_creation():
    N = 10
    sc = utils.create_subset_mask([1], N)
    Bc = utils.create_B_from_subset(sc)
    print(Bc.T)

def test_stats():

    start_time = time.perf_counter()
    for i in range(50):
        N = 20 # Measurement dimension
        M = 3 # State dimension
        NT = utils.NormalThreshold(1e-9, dof=M)

        # Create noise covariance matrix
        A = np.random.randn(N, N)
        R = A.T @ A
        L = np.linalg.cholesky(R)
        Linv = np.linalg.inv(L)

        # Create random faults
        f = 10 * np.random.randn(N, 1)

        # Generate a random measurement
        H = np.random.randn(N, M)
        x_gt = np.random.randn(M, 1)
        z = H @ x_gt + L @ np.random.randn(N,1) + f

        # Normalize measurement model
        Hnorm = Linv @ H
        znorm = Linv @ z

        # Defined Bounded Subset 
        sr = utils.create_subset_mask([], N)
        sc = utils.create_subset_mask([1], N)
        sf = utils.create_subset_mask([1,2,3,4,5,6,7,8,9,10,11,12,13], N)

        # Find solutions at bounds
        Solr = utils.mle(Hnorm, znorm, sr)
        Solc = utils.mle(Hnorm, znorm, sc)
        Solf = utils.mle(Hnorm, znorm, sf)

        # Create bounds objects
        ssbp = utils.SolutionSeparationBoundsParams(H=Hnorm, z=znorm, Solr=Solr, Solc=Solc, Solf=Solf, sr=sr, sc=sc, sf=sf, lmbda=1e-5, blanch=True)
        ssb = utils.SolutionSeparationBounds(ssbp)

        items = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        all_combos = []

        # Loop from length 1 up to the total length of the list
        for r in range(1, len(items) + 1):
            all_combos.extend(combinations(items, r))

        
        for combo in all_combos:
            sk = utils.create_subset_mask([1] + list(combo), N)
            ss_lb, ss_ub = ssb.calcBounds(sk, verbose=True)

            # Calc truth
            xk, Pk, Ik = utils.mle(Hnorm, znorm, sk)
            _,stat = NT.test(xk, Solr[0], Pk, Solr[1], lmbda=1e-5)
            # print(f"stat threshold: {NT.T}")

            print(f"ssb: {ss_lb} < {ss_ub}")

            # if NT.T < ss_lb:
            #     print("here")
            #     break
            # if NT.T < ss_ub:
            #     xk, Pk, Ik = utils.mle(Hnorm, znorm, sk)
            #     result, stat = NT.test(xk, Solr[0], Pk, Solr[1])
            #     if not result:
            #         break
            
            assert ss_ub > stat
            assert ss_lb < stat
            
        
        print(len(all_combos))
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time}")