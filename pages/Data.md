### **Accessing the data 🗃️**
Overall, TrackRAD2025 will have over 2.5 million unlabelled sagittal cine-MRI frames from 500 individual patients, and over 10,000 labelled sagittal cine-MRI frames (+8000 from frames with multiple observers) from 110 individual patients. Precisely,
a cohort of <u>500</u> unlabeled and <u>110</u> manually labeled patients has been prepared for participants. For each patient, 2D sagittal cine MRI data (time-resolved sequence of 2D images) has been acquired during the course of radiotherapy treatments at 0.35 T (ViewRay MRIdian) or 1.5 T (Elekta Unity) MRI-linacs from six international centers. Tracking targets (typically tumors) in the thorax, abdomen and pelvis were included as these can be affected by motion and reflect the most often treated anatomies on MRI-linacs. The **<u>training set</u>**, which comprises the <u>500</u> unlabeled cases plus <u>50</u> labeled cases was publicly released. Participants can further subdivide this dataset locally into training and validation. The remaining <u>60</u> labeled cases building the **<u>preliminary and final testing</u>** set is only accessible for evaluation via submission to the challenge.

Detailed information about the dataset are provided in "<u>link data paper (expected early 2025)</u>".

#### **Data location**

The training dataset can be downloaded at <u>\[link will follow\]</u> starting from March 1st, 2025.

The preliminary testing and final testing dataset are not provided to participants, but are accessible by uploading algorithms for evaluation.

#### **Data structure**

Each patient's data, both for the training and for the testing dataset, is organized using the following folder structure:

```
dataset/
|-- <patient>/
|   |-- field-strength.json
|   |-- frame-rate.json
|   |-- scanned-region.json
|   |-- images/
|   |   |-- <patient_id>_frames.mha
|   |   `-- <patient_id>_framesN.mha -> additional sequences (for some unlabelled cases)
|   `-- targets/
|       |-- <patient_id>_first_label.mha 
|       |-- <patient_id>_labels.mha
|       `-- <patient_id>_labels<observer>.mha -> additional observers
|-- D_001/
|   |-- field-strength.json -> 0.35
|   |-- frame-rate.json -> 4
|   |-- scanned-region.json -> "thorax"
|   |-- images/
|   |   `-- D_001_frames.mha
|   `-- targets/
|       |-- D_001_first_label.mha 
|       `-- D_001_labels.mha
|-- D_002/ -> this is an unlabelled case
|   |-- field-strength.json -> 0.35
|   |-- frame-rate.json -> 4
|   |-- scanned-region.json -> "thorax"
|   `-- images/
|      |-- D_001_frames.mha
|      |-- D_001_frames2.mha -> second sequence
|      `-- D_001_frames3.mha -> third sequence from the same patient
|-- D_003/
`-- .../
```

Please note that the dataset folder structure does not match the interfaces that submissions need to implement one-to-one. For details regarding submission requirements, please read the corresponding page.

#### **Data license**

Data is released under CC-BY-NC (Attribution-NonCommercial). 

#### **Data description**

This challenge provides 2D+t sagittal cine MRI data collected at six international centers:

- Amsterdam University Medical Center, Amsterdam
- Catharina Hospital, Eindhoven
- GenesisCare, Sydney
- LMU University Hospital, Munich
- Sichuan Cancer Center, Chengdu
- University Medical Center Utrecht, Utrecht

For anonymization purposes, the provenance of the data is not provided, and each center is indicated with letters from A to F. One of the centers also provided cine data with an updated MRI sequence for gating at a 1.5 T MRI-linac, this data is indicated with the letter X.

##### *Training set*

| 0.35 T MRI-linac | A       | E      |  Total |
|------------------|---------|--------|--------|
| Unlabeled        | 219     | 34     | 253    |
| Labeled          | 32?      | -      | 25?     |				

|1.5 T MRI-linac   | B       | C        | F      | Total  |
|------------------|---------|----------|--------|--------|
| Unlabeled        | 63      | 60       | 120    |   243  |
| Labeled          | 26?      | 24?      | -      |   56   |

For training, centers A, B and C provided both unlabeled and manually labeled data while centers E and F provided solely unlabeled data. 


##### *Preliminary and final testing set*

| 0.35 T MRI-linac | A       | D        |  Total |
|------------------|---------|----------|--------|
| Labeled          | 32?     | 20       | 27?     |				

|1.5 T MRI-linac   | B       | C        |  X     | Total  |
|------------------|---------|----------|--------|--------|
| Labeled          | 26?      | 24?       | 6      |   31?   |

For preliminary testing and final testing centers A,B,C, D and X provided manually labeled data.

##### *Data protocols*

###### *Unlabelled data protocol*

For the unlabeled data in the training set, cine MRI from one or multiple radiotherapy MRI-linac treatment fractions were included. Cine MRI acquired during treatment for the 0.35 T MRI-linac also include frames with degraded image quality due to gantry rotations, which participants are free to exclude using a method of their choice. 

All frames from the 1.5 T MRI-linac were acquired during treatments only and do not present degradation due to gantry rotations due to machine design.

###### *Labelled data protocol*

To avoid degraded images during evaluation, labeled frames for the 0.35 T MRI-linac were either chosen from simulation cine MRIs prior to treatment start or, when taken from treatments, a manual selection was performed to avoid periods of gantry rotation.

All frames from the 1.5 T MRI-linac were acquired during treatments only and do not present degradation due to gantry rotations due to machine design.

Human observers have generated the reference labels both for the training and testing sets. For dataset A, two observers (a medical student and a dentistry student) labeled the cine MRI frames using an in-house labeling tool developed for the challenge. For dataset B, a medical physics researcher (assistant professor) with more than 10 years experience in radiotherapy used the in-house labeling tool to delineate the frames. For dataset C, two radiation oncologists independently labeled the cine MRI frames using itk-snap. For dataset D, 4 radiation oncologists and one medical physicist have independently labeled the cine MRI frames using software provided by the 0.35 T MRI-linac vendor. 

For all labeled data, a medical physics doctoral student with 4 years experience in tumor tracking then reviewed and if necessary corrected all labels used in this challenge using the in-house tool. 

##### *Data acquisition and pre-processing* 

All images were acquired using the clinically adopted imaging protocols of the respective centers for each anatomical site and reflect typical images found in daily clinical routine. The cine-MRI sequences used at the 0.35 T and 1.5 T MRI-linacs are standardized, which ensures uniformity of the data for a given field strength. 

At the centers using the 0.35 T MRI-linac, the 2D cine-MRIs were acquired in sagittal orientation with the patient in treatment position in the MRI-linac bore. During treatment simulation or delivery, the patients performed breath holds to increase the duty cycle of the gated radiation delivery. The breath-holds are followed by periods of regular breathing. The sequence was a 2D-balanced steady-state free precession (bSSFP) at 4 Hz or 8 Hz with a slice thickness of 5, 7 or 10 mm and pixel spacing of 2.4x2.4 or 3.5x3.5 mm2.

At the centers using the 1.5 T MRI-linac, the 2D cine-MRIs were acquired in interleaved sagittal and coronal orientation with the patient in treatment position in the MRI-linac. For the challenge, only the sagittal plane has been considered. During treatment simulation or delivery, some patients performed breath holds to increase the duty cycle of the gated radiation delivery, while others breathed freely. The breath-holds are followed by periods of regular breathing. The sequence was a balanced fast ﬁeld echo (bFFE) sequence at 1 Hz to 5 Hz with a slice thickness of 5, 7 or 8 mm and pixel spacing of 1.0x1.0  to 1.7x1.7 mm2.

The following pre-processing steps were performed on the data:

- Conversion from proprietary formats to .mha
- Anonymization
- Reshaping and orientation correction
- Resampling to 1x1 mm2 in-plane pixel spacing