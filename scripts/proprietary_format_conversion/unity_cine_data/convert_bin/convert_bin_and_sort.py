#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
import SimpleITK as sitk
import datetime
import json
import struct
from collections import defaultdict
import csv
import argparse

parser = argparse.ArgumentParser(description="Convert bin to matrix, sort temporally and get jsons with additional info")
parser.add_argument("--path_input_patients", type=str, default="/your_data_in", help="Input path to folder containing patient subfolders")
parser.add_argument("--path_output_patients", type=str, default="/wyour_data_out", help="Output path where converted files are stored")
parser.add_argument("--scanned_region", type=str, default='abdomen')
parser.add_argument("--b_field_strength", type=float, default=1.5)
parser.add_argument("--patient_counter", type=int, default=1, help='Set based on how many patients have been processed already')
args = parser.parse_args()

#%%
plot=False # for debugging 

patient_mapping = {}  # Maps original patient_ID to new IF, e.g. F_XXX
cohort_string = 'F'
scan_mappings = defaultdict(dict)  # Maps patient_ID -> original scan_ID -> new scanXX
scan_counters = defaultdict(int)  # Counter for scanXX per patient

# Loops through all subdirectories of `patient_dir`, finds .bin files,
# processes them, and saves the output to `output_dir` with the required structure.
for root, _, files in sorted(os.walk(args.path_input_patients)):
    acq_times = []
    
    for file in sorted(files):
        if file.endswith(".bin"):
            bin_path = os.path.join(root, file)
            
            # Extract patient ID and scan ID from the path, specific to folder strcture!
            relative_path = os.path.relpath(bin_path, args.path_input_patients)
            path_parts = relative_path.split(os.sep)
            if len(path_parts) >= 3:
                patient_ID = path_parts[0]  # e.g., '0057857153'
                scan_ID = path_parts[2]  # e.g., '1.2.826...'
                
                # Assign new patient ID if not already assigned
                if patient_ID not in patient_mapping:
                    patient_mapping[patient_ID] = f"{cohort_string}_{args.patient_counter:03d}"
                    args.patient_counter += 1

                new_patient_ID = patient_mapping[patient_ID]
                
                # Assign new scan ID if not already assigned
                if scan_ID not in scan_mappings[patient_ID]:
                    scan_counters[patient_ID] += 1
                    scan_mappings[patient_ID][scan_ID] = f"scan{scan_counters[patient_ID]}"

                new_scan_ID = scan_mappings[patient_ID][scan_ID]

                # Define output path
                output_folder_path = os.path.join(args.path_output_patients, new_patient_ID, new_scan_ID)
                os.makedirs(output_folder_path, exist_ok=True)
                

                print(f"Original: {patient_ID}/{scan_ID} -> New: {new_patient_ID}/{new_scan_ID}")
                print(f"Processing: {bin_path}")
                # Open the binary file in read mode
                with open(bin_path, 'rb') as fid:
                    
                    # Get info where first header ends
                    raw = fid.read()  
                    beginning_of_end_first_header = raw.find(b'<Z>k__BackingField\x00\x00\x00') # That is the tag in the header (4084 is usually right after 'd')
                    if beginning_of_end_first_header != -1:
                        end_first_hader = beginning_of_end_first_header + 18 # Here you have to add 18 bytes 
                        print('...end of first header position:', end_first_hader)
                    else:
                        raise Exception('Attention: end of header not found!')
                    
                    # Get nr of rows and columns
                    text_to_find = '<Rows>k__BackingField\x18<Columns>k__BackingField\x17<Slices>k__BackingField'
                    rows_cols_position = raw.find(text_to_find.encode())
                    if rows_cols_position != -1:
                        # Move the pointer to the right position (manually found padding of 10)
                        fid.seek(rows_cols_position + len(text_to_find) + 10)  
                        
                        # Read four bytes for the row value
                        rows_data = fid.read(4)  
                        # print('Data:', rows_data)   # b'`\x01\x00\x00'
                        rows = struct.unpack('<I', rows_data)[0]  # unpack as little endian integer
                        print('Rows value:', rows)
                        
                        # Read four bytes for the column value
                        cols_data = fid.read(4)  
                        # print('Data:', cols_data)
                        columns = struct.unpack('<I', cols_data)[0]  # unpack as little endian integer
                        print('Columns value:', columns)                
                        
                    # Get voxels sizes
                    text_to_find = 'VoxelSize\x03\x00\x00\x00\x16<XInmm>k__BackingField\x16<YInmm>k__BackingField\x16<ZInmm>k__BackingField'
                    pixel_size_position = raw.find(text_to_find.encode())
                    if pixel_size_position != -1:
                        # Move the pointer to the right position (manually found padding of 10)
                        fid.seek(pixel_size_position + len(text_to_find) + 10)  
                        
                        # Read four bytes for the row value
                        pixel_size_x_data = fid.read(8)  
                        # print('Data:', pixel_size_x_data) 
                        pixel_size_x = round(struct.unpack('d', pixel_size_x_data)[0], 2)  # unpack as double precision float
                        print('Pixel size x in mm:', pixel_size_x)

                        pixel_size_y_data = fid.read(8)  
                        # print('Data:', pixel_size_y_data) 
                        pixel_size_y = round(struct.unpack('d', pixel_size_y_data)[0], 2)  # unpack as double precision float
                        print('Pixel size y in mm:', pixel_size_y)                
        
                        pixel_size_z_data = fid.read(8)  
                        # print('Data:', pixel_size_z_data) 
                        pixel_size_z = round(struct.unpack('d', pixel_size_z_data)[0], 2)  # unpack as double precision float
                        print('Pixel size z in mm:', pixel_size_z)  
                        
                    # Get info with acquisition time
                    acquisition_time_position = raw.find(b'Acquisition Time\x00\x00\x00\x00\x00\x00\x00\x00') # That is the tag in the header
                    if acquisition_time_position != -1:
                        print('...acquisition time position:', acquisition_time_position)
                        fid.seek(acquisition_time_position + 24) # Here you have to add 24 bytes to get to the actual position of the time
                        time_data = fid.read(8) 	# From here on you read 8 bytes
                        time_data_unsign_int = int.from_bytes(time_data, 'little', signed=False) # You need to convert into integers (data is little endian and unsigned)
                        
                        acquisition_time_in_sec = round(time_data_unsign_int / 10.**7, 3)	# You get an integer number, that is an Windows SQL time format in seconds, rounded to the millisecond
                        print(f'...acquisition time in sec: {acquisition_time_in_sec}')                
                        acquisition_time_in_UXT = datetime.datetime(1900,1,1) + datetime.timedelta(seconds=acquisition_time_in_sec) + datetime.timedelta(hours=1)	# Here convert into a readable timestamp and add one hour for CET in Germany.
                        print(f'...acquisition time in UXT: {acquisition_time_in_UXT}')
                    acq_times.append(acquisition_time_in_sec) 
                    
                        
                    # Start reading file back from the beginning
                    fid.seek(0)
                    
                    # Dump the header 
                    header = np.fromfile(fid, dtype=np.int8, count=end_first_hader)

                    # Read the following image data
                    image = np.fromfile(fid, dtype=np.uint16, count=rows*columns)
                    image_reshaped = np.reshape(image, (rows, columns))
                    
                    if plot:
                        # Plot the current image   
                        plt.figure()
                        plt.title('Reshaped only')
                        plt.imshow(image_reshaped, cmap='gray', vmin=0, vmax=1000)
                        plt.colorbar()
                        plt.axis('image')
                        # plt.savefig('/workspace/tmp/tmp_original.png')

                    # Images need to be shifted to be centered
                    shift_amount = 22
                    shifted_image = shift(image_reshaped, (0, -shift_amount), mode='constant')
                    # Get pixels back which were shifted outside of boundary
                    shifted_image[:, -shift_amount:] = image_reshaped[:, :shift_amount]

                    if plot:
                        plt.figure()
                        plt.title('Shifted')
                        plt.imshow(shifted_image, cmap='gray', vmin=0, vmax=1000)
                        plt.colorbar()
                        plt.axis('image')
                        # plt.savefig('/workspace/tmp/tmp_shifted.png')
            
                    
                    # Check if there is nan in image array
                    if np.isnan(np.min(shifted_image)):
                        raise Exception ('NaN found in extracted image!')
                    
                    
                    # Create filename for saving
                    file_name_out = f'frame_{acquisition_time_in_sec:.3f}_{file[0:-4]}.mha'
                    path_to_output_file = os.path.join(output_folder_path, file_name_out)
                    
                    # Save as mha with pixel size info and third dim (h,w,1)
                    sitk_image = sitk.GetImageFromArray(shifted_image[None,:,:].astype(np.float32)) 
                    sitk_image.SetSpacing([pixel_size_y, pixel_size_x, pixel_size_z])
                    sitk.WriteImage(sitk_image, path_to_output_file)
                    print('...saved as mha:', path_to_output_file)


    # For current scan, calculate differences between consecutive numbers 
    if len(acq_times) > 0:
        differences = np.diff(sorted(acq_times))

        # Calculate median of differences
        median_diff = np.median(differences)
        print(f'Median difference between consecutive frames: {median_diff} s')
        # Save acquisiiton frequency divided by three (only sagittal) as json
        freq_sag = 1/median_diff/3
        with open(os.path.join(output_folder_path, 'frame-rate.json'), 'w') as f:
            json.dump(round(freq_sag,2), f, indent=4)

        if plot:
            # Plot the histogram of the differences
            plt.hist(differences, bins=20, color='blue', alpha=0.7, edgecolor='black')
            plt.title("Histogram of Differences Between Consecutive Times")
            plt.xlabel("Difference (seconds)")
            plt.ylabel("Frequency")
            plt.grid(axis='y', alpha=0.75)
            plt.show()  
            # Re-bin the data for better resolution within the 1-second range
            plt.hist(differences, bins=np.linspace(0, 1, 21), color='blue', alpha=0.7, edgecolor='black')  # 20 bins within 1 second
            plt.title("Histogram of Differences Between Consecutive Times (Up to 1 Second)")
            plt.xlabel("Difference (seconds)")
            plt.ylabel("Frequency")
            plt.xlim(0, 1)  # Limit x-axis to 1 second
            plt.grid(axis='y', alpha=0.75)
            plt.show()
            
        # Save anatomical site and b-field
        with open(os.path.join(output_folder_path, 'scanned-region.json'), 'w') as f:
            json.dump(args.scanned_region, f, indent=4)
            
        with open(os.path.join(output_folder_path, 'b-field-strength.json'), 'w') as f:
            json.dump(args.b_field_strength, f, indent=4)
              
 
csv_file = os.path.join(args.path_output_patients, f"patient_scan_mappings_{os.path.basename(args.path_input_patients)}.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Original Patient ID", "New Patient ID", "Original Scan ID", "New Scan ID"])
    for patient_ID, new_patient_ID in patient_mapping.items():
        for original_scan_ID, new_scan_ID in scan_mappings[patient_ID].items():
            writer.writerow([patient_ID, new_patient_ID, original_scan_ID, new_scan_ID])
print(f"Saved patient mappings to {csv_file}.")
