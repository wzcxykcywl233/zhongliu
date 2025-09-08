from pathlib import Path
from datetime import datetime
import numpy as np
import time

from inference import (
    load_json_file,
    load_image_file_as_array,
    write_array_as_image_file,
)

PATH_DATASET = Path("/home/felix/trackrad-cotracker-submission/dataset/labeled")


def run_inference_for_case(input_path: Path, output_path: Path):
    print(f"Running inference for case: {input_path}")
    loading_start_time = time.perf_counter()

    # Read the inputs
    input_frame_rate = load_json_file(
        location=input_path / "frame-rate.json",
    )
    input_magnetic_field_strength = load_json_file(
        location=input_path / "b-field-strength.json",
    )
    input_scanned_region = load_json_file(
        location=input_path / "scanned-region.json",
    )
    input_mri_linac_series = load_image_file_as_array(
        location=input_path / "images",
    )

    input_mri_linac_target = load_image_file_as_array(
        location=input_path / "targets",
        filename_contains="first",
    )

    print(f"Runtime loading:   {time.perf_counter() - loading_start_time:.5f} s")

    from model import run_algorithm

    algo_start_time = time.perf_counter()

    output_mri_linac_series_targets = run_algorithm(
        frames=input_mri_linac_series,
        target=input_mri_linac_target,
        frame_rate=input_frame_rate,
        magnetic_field_strength=input_magnetic_field_strength,
        scanned_region=input_scanned_region,
    )

    # Enforce uint8 as output dtype
    output_mri_linac_series_targets = output_mri_linac_series_targets.astype(np.uint8)

    print(f"Runtime algorithm: {time.perf_counter() - algo_start_time:.5f} s")

    writing_start_time = time.perf_counter()

    # Save the output
    write_array_as_image_file(
        location=output_path / "images/mri-linac-series-targets",
        array=output_mri_linac_series_targets,
    )
    print(f"Runtime writing:   {time.perf_counter() - writing_start_time:.5f} s")


if __name__ == "__main__":
    exp_dir = Path("/home/felix/trackrad-cotracker-submission/.results/")
    exp_dir = exp_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir.mkdir(parents=True, exist_ok=True)

    for case_folder in PATH_DATASET.iterdir():
        if not case_folder.is_dir() or case_folder.name.startswith("."):
            continue
        input_path = Path(case_folder)
        output_path = exp_dir / case_folder.name
        run_inference_for_case(input_path, output_path)
