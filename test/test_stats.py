import ssfde
import utils
import numpy as np

def test_B_creation():
    N = 10
    sc = utils.create_subset_mask([1], N)
    Bc = utils.create_B_from_subset(sc)
    print(Bc.T)

def test_stats():

    for i in range(1000):
        N = 10 # Measurement dimension
        M = 3 # State dimension
        NT = utils.NormalThreshold(1e-9, dof=M)

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

        # Defined Bounded Subset 
        sr = utils.create_subset_mask([], N)
        sc = utils.create_subset_mask([1,2,3], N)
        sf = utils.create_subset_mask([1,2,3,4,5,6], N)

        # Find solutions at bounds
        Solr = utils.mle(Hnorm, znorm, sr)
        Solc = utils.mle(Hnorm, znorm, sc)
        Solf = utils.mle(Hnorm, znorm, sf)

        print(f"Pc: {Solc[1]}")
        print(f"{np.linalg.eigvals(Solc[1])}")

        # Create bounds objects
        ssbp = utils.SolutionSeparationBoundsParams(H=Hnorm, z=znorm, Solr=Solr, Solc=Solc, Solf=Solf, sr=sr, sc=sc, sf=sf)
        ssb = utils.SolutionSeparationBounds(ssbp)

        sk = utils.create_subset_mask([1,2,3,4], N)
        ss_lb, ss_ub = ssb.calcBounds(sk, 1)

        # Calc truth
        xk, Pk, Ik = utils.mle(Hnorm, znorm, sk)
        print(f"Pk: {Pk}")
        print(f"{np.linalg.eigvals(Pk)}")
        _,stat = NT.test(xk, Solr[0], Pk, Solr[1])

        print(f"ssb: {ss_lb} < {stat} < {ss_ub}")
        assert ss_lb < stat
        assert ss_ub > stat

        # xr = Solr[0]
        # Dinv = np.diag([1e-8, 1, 1e-8])
        # Dsqrt = np.sqrt(np.linalg.inv(Dinv))
        # Dinvsqrt = np.sqrt(Dinv)
        # norm = np.linalg.norm(Dinvsqrt @ (xk - xr))**2
        # Bk = utils.create_B_from_subset(sk)
        # Hk = Bk.T @ Hnorm
        # zk = Bk.T @ znorm

        # dp = np.linalg.norm(Dinvsqrt @ Hk.T @ (zk - Hk @ xr))**2
        # QI = Dinvsqrt @ Hk.T @ Hk @ Dsqrt
        # lb = dp / np.linalg.norm(QI, ord=2)**2
        # Q = Dinvsqrt @ np.linalg.inv(Hk.T @ Hk) @ Dsqrt
        # ub = dp * np.linalg.norm(Q, ord=2)**2
        # print(np.trace(Q))
        # print(np.trace(Q.T @ Q))
        # print(np.linalg.eigvals(Q.T @ Q))
        # print(f"{lb} < {norm} < {ub}")
        # assert lb < norm
        # assert ub > norm

    # print(utils.calc_ss_statistic_baseline(s1, s2, Hnorm, znorm, NT))
    # print(utils.calc_ss_statistic_ours(s1, s2, Hnorm, znorm, NT))