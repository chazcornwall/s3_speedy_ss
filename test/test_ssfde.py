import numpy as np
import matplotlib.pyplot as plt
from ssfde import SSFDEParams, SSFDE, NormalThreshold, mle
from utils import BaselineTest, NormalThreshold, random_measurement_model_with_faults, BlanchTest, OurTest, RandomMeasurementModel, create_subset_mask, create_B_from_subset

Ntrials = 100
N = 30 # Measurement dimension 30
Nf = 7 # 7
M = 3 # State dimension
fault_bias = (-100, 100)
NT = NormalThreshold(1e-10, dof=M)


def test_ssfde_baseline():

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z, heuristic="ours")
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M, max_duration_s=40)
    if output:
        print(f"Time (s): {output[2]:0.5f}")
        print(f"Fault-free subset: {output[0]}")
        print(f"Truth Faults: {f_indices}")
        for i in range(N):
            if i in f_indices:
                assert not output[0][i]

def test_ssfde_blanch():

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=BlanchTest, NT=NT, H=H, z=z)
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M, verbose=False, max_duration_s=10)
    if output:
        print(f"Time (s): {output[2]:0.5f}")
        print(f"Fault-free subset: {output[0]}")
        print(f"Truth Faults: {f_indices}")
        for i in range(N):
            if i in f_indices:
                assert not output[0][i]
            else:
                assert output[0][i]

def test_ssfde_ours():

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=OurTest, NT=NT, H=H, z=z)
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M, verbose=False, max_duration_s=5)
    if output:
        print(f"Time (s): {output[2]:0.5f}")
        print(f"Fault-free subset: {output[0]}")
        print(f"Truth Faults: {f_indices}")
        for i in range(N):
            if i in f_indices:
                assert not output[0][i]
            else:
                assert output[0][i]

def test_ssfde_all():
    METHODS = ["Baseline", "Blanch", "Ours", "Omnicient"]
    errors = [np.zeros((M, Ntrials)), np.zeros((M, Ntrials)), np.zeros((M, Ntrials)), np.zeros((M, Ntrials))]
    times_elapsed = np.zeros((len(METHODS), Ntrials))

    rmm = RandomMeasurementModel(N, M, Nf, fault_bias)

    for n in range(Ntrials):
        H, z, x_gt, f_indices = rmm.sample()

        params_baseline = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z)
        ssfde_object_baseline = SSFDE(params_baseline)

        params_blanch = SSFDEParams(TestStatClass=BlanchTest, NT=NT, H=H, z=z)
        ssfde_object_blanch = SSFDE(params_blanch)

        params_our = SSFDEParams(TestStatClass=OurTest, NT=NT, H=H, z=z)
        ssfde_object_our = SSFDE(params_our)

        ssfdes = [ssfde_object_baseline, ssfde_object_blanch, ssfde_object_our]
        for i, ssfde_object in enumerate(ssfdes):
            _, sol, time_elapsed = ssfde_object.fde(N, M, verbose=False, max_duration_s=0.2)
            errors[i][:,[n]] = sol[0] - x_gt
            times_elapsed[i][n] = time_elapsed

        # Calculate error knowing where faults are for perspective
        sk = create_subset_mask(f_indices, N)
        sol_perfect = mle(H, z, sk)
        errors[3][:,[n]] = sol_perfect[0] - x_gt

    
    # Create three horizontal subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    min_time = np.min(times_elapsed)
    max_ss_stat = max(np.max(np.linalg.norm(errors[0], axis=0)), np.max(np.linalg.norm(errors[1], axis=0)), np.max(np.linalg.norm(errors[2], axis=0)))

    # Plot histograms
    for i, ax in enumerate(axes):
        axes[i].hist(np.linalg.norm(errors[i], axis=0), bins=30)
        ax.set_xlabel("SS Stat [m]")
        ax.set_ylabel("Frequency [Counts]")
        ax.set_xlim(0, max_ss_stat)
        ax.set_title(METHODS[i])
        ax.grid(True, alpha=0.3)
        print(f"{METHODS[i]} RMSE: {np.sqrt(np.mean(np.linalg.norm(errors[i], axis=0)**2))}")
        print(f"{METHODS[i]} Mean Time Elapsed: {np.mean(times_elapsed[i,:])} s")

    plt.tight_layout()
    plt.show()

    