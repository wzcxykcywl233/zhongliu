"""
Edit this file to implement your algorithm. 

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
"""
from pathlib import Path
import numpy as np

RESOURCE_PATH = Path("resources")  # load weights and other resources from this directory


def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    """
    Implement your algorithm here.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
    - target (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
    - frame_rate (float): The frame rate of the MRI linac series.
    - magnetic_field_strength (float): The magnetic field strength of the MRI linac series.
    - scanned_region (str): The scanned region of the MRI linac series.
    """
    
    # frames.shape == (W, H, T)
    # target.shape == (W, H, 1)

    # For the example we want to repeat the initial segmentation for every frame 
    repeated_target = np.repeat(target, frames.shape[2], axis=-1)

    # repeated_target.shape == (W, H, T)
    return repeated_target
