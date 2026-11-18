import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ssfde import SSFDEParams, SSFDE, NormalThreshold, mle
from utils import BaselineTest, NormalThreshold, random_measurement_model_with_faults, BlanchTest, OurTest, RandomMeasurementModel, create_subset_mask, create_B_from_subset, get_output

plt.rcParams['font.size'] = 16

Ntrials = 100
N = 15 # Measurement dimension 30
Nf = 7 # 7
M = 4 # State dimension
T_MAX = 1.0 # Max time
NUM_TIME_STEPS = 7
fault_bias = (-100, 100)
NT = NormalThreshold(1e-10, dof=M)
LOAD_DATA = False
ERROR_FILE_NAME = "test_ssfde_all_20261112_110903_errors.npy"
TIMES_FILE_NAME = "test_ssfde_all_20261112_110903_times.npy"


def test_ssfde_baseline():

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z, heuristic="num")
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M+1, max_duration_s=20)
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

    output = ssfde_object.fde(N, M+1, verbose=False, max_duration_s=10)
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

    output = ssfde_object.fde(N, M+1, verbose=False, max_duration_s=5)
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
    METHODS = ["Baseline", "Chi2", "Ours", "Omniscient"]

    now = datetime.now()
    filestamp = now.strftime("%Y%m%d_%H%M%S")
    if not LOAD_DATA:
        times = np.linspace(0, T_MAX, NUM_TIME_STEPS)

        errors = []
        for i in range(len(METHODS)):
            errors.append(np.zeros((M, Ntrials, NUM_TIME_STEPS)))

        rmm = RandomMeasurementModel(N, M, Nf, fault_bias)

        for n in range(Ntrials):
            H, z, x_gt, f_indices = rmm.sample()

            params_const = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z, heuristic="const")
            ssfde_object_const = SSFDE(params_const)

            params_chi2 = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z, heuristic="chi2")
            ssfde_object_chi2 = SSFDE(params_chi2)

            params_our = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z, heuristic="num")
            ssfde_object_our = SSFDE(params_our)

            ssfdes = [ssfde_object_const, ssfde_object_chi2, ssfde_object_our]
            for i, ssfde_object in enumerate(ssfdes):
                output_list = ssfde_object.fde(N, M+1, verbose=False, max_duration_s=T_MAX)
                for j, t in enumerate(times):
                    _, sol, time_elapsed = get_output(t, output_list)
                    errors[i][:,[n],j] = sol[0] - x_gt

            # Calculate error knowing where faults are for perspective
            sk = create_subset_mask(f_indices, N)
            sol_perfect = mle(H, z, sk)
            for j in range(len(times)):
                errors[3][:,[n],j] = sol_perfect[0] - x_gt

        # Save data output
        dir = "data/"
        filename = "test_ssfde_all" + "_" + filestamp
        np.save(dir + filename + "_errors.npy", errors)
        np.save(dir + filename + "_times.npy", times)
        np.save(dir + filename + "_Hnorm.npy", rmm.Hnorm)
        np.save(dir + filename + "_f.npy", rmm.f)
    else:
        errors = np.load("data/" + ERROR_FILE_NAME)
        times = np.load("data/" + TIMES_FILE_NAME)
    
    # Create three horizontal subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    max_rmse = 0.0
    for i in range(len(METHODS)):
        curr_max_rmse = np.max(np.linalg.norm(errors[i], axis=0))
        if curr_max_rmse > max_rmse:
            max_rmse = curr_max_rmse

    # Plot histograms
    violion_plot_width = T_MAX/(2.25 * NUM_TIME_STEPS)
    handles, labels = [], []
    for i, ax in enumerate(axes):
        # axes[i].hist(np.linalg.norm(errors[i], axis=0), bins=30)
        vp = ax.violinplot(np.linalg.norm(errors[i], axis=0), times, showmeans=True, widths=violion_plot_width)
        ax.set_xlabel("Time [s]")
        if i == 0:
            ax.set_ylabel("Root Squared Error [m]")
        ax.set_xlim(-violion_plot_width, T_MAX+violion_plot_width)
        ax.set_ylim(0, max_rmse + 0.1)
        ax.set_title(METHODS[i])
        ax.grid(True, alpha=0.3)
        print(f"{METHODS[i]} RMSE: {np.sqrt(np.mean(np.linalg.norm(errors[i], axis=0)**2))}")
        h, l = ax.get_legend_handles_labels()
        handles.extend(h + [vp['bodies'][0]])
        labels.extend(l + ['Pred Error'])

    fig.legend(
        handles=[handles[-1]],
        labels=[labels[-1]],
        loc="upper center",          # Alignment point of the legend box
        bbox_to_anchor=(0.5, 0.1), # Coordinates relative to the figure (x, y) # bbox_to_achor=(0.5, 0.05) when creating introduction image
        ncols=2                      # Make the legend layout horizontal
    )

    plt.tight_layout()
    plt.savefig("data/" + "test_ssfde_all" + "_" + filestamp + "_fig.png")
    plt.show()
    

    