# Conversion of Unity binary data
This repository contains code to convert Unity cine MRI data from the proprietary .bin format into .mha. When these .bin files are exported from a Unity, they usually end up in a folder structure like /patient_ID/1/1.2.82.../TwoDImages/*.bin where each .bin file is one 2D frame.
The script `convert_bin_and_sort.py` reads in those .bin, gets some information such as image size, time stamp of the frame, etc. from the header and then reads out the image bytes and saves them as .mha. The script `classify_orientation_cnn_pxl.py` then uses a convolutional nueral network which was explicitly trained for this task to classify the .mha files into axial, coronal and sagittal orientation. As this classification works for about 99% of the frames, an additional `clean_up_misclassified.py` script was written which uses a pixel encofing in the top left cornen of each frame to see whether the pixel encoding in one folder is the same for all frames. Unfortunately this encoding could not be used to classify the orientation in the first place as it is different for every scan.
For TrackRAD2025 the data was abundant so we simply deleted the scans for which the classification did not work. An alternative approach could be to move the misclassfied frames to the correct orientation folder based on the pixel values in the top left corner or to re-train the convolutional neural netowrk with the correctly classifed data.

Running the code:
* Build a Docker image based on the provided Dockerfile and run a container mounting the folder `unity_cine_data` to for instance the folder `/workspace` in the container.
* `main.sh`: Set the `p1`, `p2`, `p6` and `p7` path variables and run this bash script to convert, sort and then classify the .bin image data into .mha.
* `clean_up_misclassified.py`: If needed, set the `base_dir` and `output_dir` variables and then run this python script to clean up the misclassified scans.





