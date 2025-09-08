# Conversion of MRIdian binary data
This repository contains code to convert MRIdian cine MRI data from the proprietary .dat or .raw format into .mha.
* `convert_dat`: MATLAB code to convert cine MRI frames from .dat format into .mha. Based on ConvertSendExternalToMatlabVolume.m script provided by Viewray. These .dat files can be obtained from .dump files which can be found locally on the treatment console.
	* Set the variable `path_data` in the `main_convert_cine.m` script to the folder with the .dat file to convert the images to .mha.
* `convert_raw_and_roi`: Matlab code to convert .raw imaging format into .mha  and C++ code to convert target tracking contour coordinates from binary into .txt. Based on read-roi.cpp script provided by Viewray and on image and contour data provided by Viewray.
	* For raw conversion:
		* Set the variable `path_to_input_raw` in the `main_convert_raw.m` script to your .raw image file to be converted and run the script to convert the image to .mha.
	* For roi conversion:		
		* Build a Docker image based on the provided Dockerfile and run a container mounting the folder `convert_raw_and_roi` to for instance the folder `/workspace` in the container.
		* Run the `build_read_roi.sh` script to compile the cpp code.
		* Set the path to your ROI file and run the `execute_read_roi.sh` script to read and convert to coordinates in a .txt.


