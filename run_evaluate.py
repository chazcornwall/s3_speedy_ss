import argparse
from truth.recorded_truth import RecordedTruth
from measurements.recorded_measurement import RecordedMeasurement
from measurements.generated_measurement import GeneratedMeasurement, GeneratedMeasurementParams
from utils.evaluate import evaluate
from ssfde import NormalThreshold, SSFDE, SSFDEParams
from utils.utils import BaselineTest


URBAN_NAV_HARSH_GT = "recorded_data/UrbanNav_HK_Harsh_Urban_GT.txt"
URBAN_NAV_HARSH = "recorded_data/UrbanNav_HK_Harsh_Urban_Input.txt"


def main():

    # Create Truth and Measurement objects!
    rt = RecordedTruth(URBAN_NAV_HARSH_GT)
    rm = RecordedMeasurement(URBAN_NAV_HARSH)
    gm_params = GeneratedMeasurementParams(stddev=10.0, fault_prob=0.5, fault_magnitude=0)
    gm = GeneratedMeasurement(rt, rm, gm_params)
    NT = NormalThreshold(1e-10, dof=3) # Ignore bias term!

    def create_our_ssfde(H, z):
        params_our = SSFDEParams(TestStatClass=BaselineTest, NT=NT, H=H, z=z, heuristic="num")
        ssfde_object_our = SSFDE(params_our)
        return ssfde_object_our


    # time_range=(15,20) for a single detected fault (5s max duration)
    # time_range=(500, 504) for a double detected fault (5s max duration)
    # time_range=(853, 855) for no detected fault (50s max duration)
    evaluate(rt, rm, [create_our_ssfde], time_range=(500, 504), max_ssfde_duration_s=5.0)


if __name__ == "__main__":
    main()