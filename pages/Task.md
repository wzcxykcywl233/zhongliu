### **Task**

TrackRAD2025 will consist of a single task. The task will be real-time tumor tracking on time-resolved sagittal 2D cine-MRI sequences. Algorithms will be provided with a template tumor segmentation on the first frame, and the remaining 2D cine-MRI sequence requiring real-time segmentation. 
bAn example of the output the algorithms should produce (a tumor segmentation mask on each frame) can be seen in the video below:

<video src="https://github.com/LMUK-RADONC-PHYS-RES/trackrad2025/raw/refs/heads/main/images/GIF_overlay_preds_mediastinum02.mp4" style="width: 50%" controls muted autoplay></video>

In summary:

**Inputs** 

- 1 cine-MRI frame with template tumor segmentation mask

- Remaining sequence of unlabelled cine-MRI frames

**Outputs**

- Tumor segmentation masks on the unlabelled cine-MRI frames

**Attention!** To enable real-time applications, your algorithm may not use information from later frames to produce the masks for previous frames.

<br/>

### Getting started

Follow these steps to set up your development environment and prepare your submission.

#### 1. Clone the Official Repository
Begin by cloning the official [trackrad2025 repository](https://github.com/LMUK-RADONC-PHYS-RES/trackrad2025):

```bash
git clone git@github.com:LMUK-RADONC-PHYS-RES/trackrad2025.git
```

This repository includes:

- Challenge information
- Code used for evaluation and local testing
- A baseline algorithm to help you get started

#### 2. Download and Place the Dataset

Download the [dataset](https://trackrad2025.grand-challenge.org/data/) from Zenodo and place it in the dataset/ folder. Use the labeled dataset for supervised training and model evaluation. For unsupervised training you may want to use the unlabeled dataset, for supervised training and evaluation of your model the labeled dataset is sufficient.

 
> Tip: A synthetic sequence (formatted like the labeled data) is also provided in the repository to help you get started quickly, even while still downloading the dataset.

#### 3. Test the Baseline Algorithm
Before modifying anything, verify that the baseline and evaluation are working correctly:

```
sh test-algorithm.sh
```

This script will compile, execute, and test the baseline algorithm with the synthetic example scan. 

Familiarize yourself with the parameters at the top of the script (especially ALGORITHM_DIR and DATASET_DIR). You will need to update these paths when evaluating your own algorithm.

#### 4. Understand the baseline

Open `baseline-algorithm/model.py` in your code editor. You will see the following function:
 
```python
def run_algorithm(frames: np.ndarray, 
    target: np.ndarray, 
    frame_rate: float, 
    magnetic_field_strength: float, 
    scanned_region: str) -> np.ndarray:
    """
    Implement your algorithm here.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
    - target (numpy.ndarray): A 2D numpy array of shape (W, H) containing the MRI linac target.
    - frame_rate (float): The frame rate of the MRI linac series.
    - magnetic_field_strength (float): The magnetic field strength of the MRI linac series.
    - scanned_region (str): The scanned region of the MRI linac series.
    """
    
    # frames.shape == (W, H, T)
    # target.shape == (W, H)

    # For the example we want to repeat the initial segmentation for every frame 
    repeated_target = np.repeat(target, frames.shape[2], axis=-1)

    # repeated_target.shape == (W, H, T)
    return repeated_target
```


#### 5. Start Developing Your Own Model

Copy the baseline algorithm folder to a new folder to work in.

```bash
cp -R baseline-algorithm your-algorithm
```

Navigate into your new folder and initialise it as a separate git repository:

```bash
cd your-algorithm
git init
```

Modify the test-algorithm.sh script in the main repository to point to your new folder (update the ALGORITHM_DIR path).

```bash
ALGORITHM_DIR="./your-algorithm"
```

> Tip: By this time your download of the labeled dataset could be done and you could also update the dataset path, to use the labeled dataset. If you don't just want to verify that your algorithm works on a technical level, that is. 
 
```bash
# This might also be a good time to 
DATASET_DIR="./dataset/labeled"
```

#### 6. Implement Your Model:

Edit the model.py file in your working copy to implement your algorithm. Make sure to keep the function signature of the run_algorithm function intact. This way you ensure that your submission is functional.

```python
def run_algorithm(frames: np.ndarray, # frames.shape == (W, H, T)
    target: np.ndarray, # target.shape == (W, H)
    frame_rate: float, 
    magnetic_field_strength: float, 
    scanned_region: str) -> np.ndarray:

    # Example: Repeat the initial segmentation 
    repeated_target = np.repeat(target, frames.shape[2], axis=-1)

    return repeated_target # repeated_target.shape == (W, H, T)
```

**Note:** When adding files or folders to it make sure to add them the the Dockerfile. This is required for the submission on grand-challenge, where you provide the platform access to github repository you created earlier. Alternatively, but not recommended, you can upload a container image, e.g. created using the save.sh script provided with the baseline algorithm. 
**However, we still require read access to your model source code for price eligibility.**

#### 7. Have Fun

Have fun developing and contributing the the future of cine-MRI target tracking!

P.S.: When you are ready to test your submission on grand-challenge or submit your final version, go to the [Submission Instructions](https://trackrad2025.grand-challenge.org/submission-instructions/).