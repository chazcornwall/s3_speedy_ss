import numpy as np
from ssfde import SSFDEParams, SSFDE, NormalThreshold
from utils import BaselineTest, NormalThreshold, random_measurement_model_with_faults, BlanchTest, OurTest



def test_ssfde_baseline():
    N = 10 # Measurement dimension
    Nf = 3
    M = 3 # State dimension
    fault_bias = (-100, 100)
    NT = NormalThreshold(1e-9, dof=M)

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z)
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M)
    if output:
        print(f"Time (s): {output[2]:0.5f}")
        print(f"Fault-free subset: {output[0]}")
        print(f"Truth Faults: {f_indices}")
        for i in f_indices:
            assert not output[0][i]

def test_ssfde_blanch():
    N = 10 # Measurement dimension
    Nf = 3
    M = 3 # State dimension
    fault_bias = (-100, 100)
    NT = NormalThreshold(1e-9, dof=M)

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=BlanchTest, NT=NT, H=H, z=z)
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M, verbose=True)
    if output:
        print(f"Time (s): {output[2]:0.5f}")
        print(f"Fault-free subset: {output[0]}")
        print(f"Truth Faults: {f_indices}")
        for i in f_indices:
            assert not output[0][i]

def test_ssfde_ours():
    N = 10 # Measurement dimension
    Nf = 3
    M = 3 # State dimension
    fault_bias = (-100, 100)
    NT = NormalThreshold(1e-9, dof=M)

    H, z, x_gt, f_indices = random_measurement_model_with_faults(N, Nf, M, fault_bias)

    params = SSFDEParams(TestStatClass=OurTest, NT=NT, H=H, z=z)
    ssfde_object = SSFDE(params)

    output = ssfde_object.fde(N, M, verbose=True)
    if output:
        print(f"Time (s): {output[2]:0.5f}")
        print(f"Fault-free subset: {output[0]}")
        print(f"Truth Faults: {f_indices}")
        for i in f_indices:
            assert not output[0][i]