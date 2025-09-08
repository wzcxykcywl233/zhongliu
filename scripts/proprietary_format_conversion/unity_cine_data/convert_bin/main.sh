#!/bin/bash

path_to_code="/your_loca_path/proprietary_format_conversion/unity_cine_data/convert_bin"  # path in docker container where code is stored

echo "---------------Starting conversion of bin files---------------"
p1="/your_local_path/proprietary_format_conversion/unity_cine_data/convert_bin/data_test/your_patient_folder"   # path to folder with input patients in .bin
p2="/your_output_folder"   # path to folder with output .mha 
p3="abdomen"  # anatomical site to be written in the output patient folder
# p3="pelvis"  # anatomical site to be written in the output patient folder
# p3="thorax"  # anatomical site to be written in the output patient folder
p4=1.5   # field strength of MRI-linac
p5=1  # patient counter, can be manually increased if script is run multiple times for different input patient folders
# p5=13  # patient counter, can be manually increased if script is run multiple times for different input patient folders
command1="python $path_to_code/convert_bin_and_sort.py --path_input_patients $p1 --path_output_patients $p2 --scanned_region $p3 --b_field_strength $p4 --patient_counter $p5"
$command1

echo "---------------Starting classfication of orientations---------------"
p6="/your_output_folder"  # path to folder with all .mha
p7="/your_output_folder_by_orientation"  # path to folder with .mha files classifed by orientation (axial, coronal, sagittal)
p8=$path_to_code/orientation_model/cnn_orientation_model_0.0016036431093155962.pth   # path to neural network used for classification
command2="python $path_to_code/classify_orientation_cnn_pxl.py --path_input_patients $p6 --path_output_patients $p7 --path_classifier $p8"
$command2

echo "Done!"
