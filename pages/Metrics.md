### Metrics 📊

Submitted algorithms/models will be evaluated by comparing predicted target segmentations against ground truth labels. For this purpose, 4 geometric and one radiotherapy-specific dose metric are used. Moreover the runtime of the algorithm per cine-MRI frame will be considered.

The specific implementation of the evaluation can be found in the [TrackRad2025 git repository](https://github.com/LMUK-RADONC-PHYS-RES/trackrad2025).

##### Geometric metrics

For each individual cine-MRI frame, the model predictions will be evaluated with the following geometric metrics:

- The [Dice similarity coefficient](https://metrics-reloaded.dkfz.de/metric?id=dsc) (DSC) between the model prediction and the ground truth.
- The surface distance distribution between the model prediction and the ground truth. Here the 95th percentile of the distribution will be employed. This is also often called the [95th percentile Hausdorff distance](https://metrics-reloaded.dkfz.de/metric?id=hd95) (HD95).
- The [mean average surface distance](https://metrics-reloaded.dkfz.de/metric?id=masd) (MASD) between the model prediction and the ground truth.
- The Euclidean [center distance](https://metrics-reloaded.dkfz.de/metric?id=center_distance) (CD) of the center of mass of the model prediction and the ground truth.

##### Dose metric

For each cine-MRI sequence, the model prediction will be evaluated with a radiotherapy-specific dose metric. This dose metric estimates the accuracy of a radiotherapy dose delivery using multileaf collimator tracking based on the model predictions. 

Please note that any model optimising for the geometric metrics above is also optimising for this dose metric.

To compute the dose metric, the ground truth label of the first frame will be converted to an approximated radiation therapy dose by applying a 3 mm expansion of the gross tumor volume (GTV) indicated by the label to the clinical target volume (CTV). 
Subsequently, this expanded mask will be smoothed by a Gaussian of 6 mm standard deviation for targets in the lung (simulates a dose fall-off similar to those observed in clinical dose distributions for lung patients) and of 4 mm for all other targets (dose fall-off for targets in higher density tissue). This dose distribution will then be shifted by the distance between the ground truth centroid position of the tracking target and the centroid position obtained by the investigated model for each frame. These shifted distributions will be averaged to get a centroid-error shifted dose. The relative difference between the GTV (or tracking target) D98% (from the cumulative dose volume histogram) for the ground truth dose distribution and the final shifted dose distribution will be calculated for each patient.

#### Run-time
For each algorithm, the runtime per cine-MRI frame will be evaluated. Algorithms exceeding a maximum runtime of 1 sec per frame (plus model and data loading time) on the provided hardware will be excluded from the challenge due to concerns for real-time applicability of the algorithm.

#### Metrics aggregation

For each geometric metric, the obtained results per cine frame will be averaged over all frames of a given cine-MRI first. The average over all test cases will then be determined to obtain a single value per geometric metric and model. 

For the dose metric results per cine-MRI sequence will be averaged over all test cases.

The model inference time per frame will be computed for all cases and divided by the total number of frames, minus model and data loading overhead, estimated using a linear model.

#### Ranking

Submissions are first ranked per individual metric (6 ranks). For each submission, the rank for each average metric will be calculated compared to all submissions with the following ordering:

- DSC: higher is better
- HD95: lower is better
- MASD: lower is better
- CD: lower is better
- Dose metric/relative D98%: higher is better
- Inference time per frame: lower is better

The final rank for a submission is obtained by computing the average rank over the 6 metrics, ranging from 1 (best submission) to n (worst submission). 


#### Minimal requirements

Submissions are expected to outperform a simple baseline model mimicking a no-tracking approach, i.e., copying the ground truth label of the first frame to all other frames. Teams will not be considered in the ranking if their method does not outperform the no-registration baseline in at least one of DSC, HD95, MASD or CD.

To enforce real-time speeds on available hardware, a time-limit it set for each run (1 sec inference time per frame plus overhead due to model and data loading).