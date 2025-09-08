#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import shutil
from collections import Counter
import SimpleITK as sitk
import tqdm
import torch
from monai.transforms import Compose, LambdaD, EnsureChannelFirstD, LoadImageD, RotateD, CenterSpatialCropD, ToTensorD, ScaleIntensityRangeD, FlipD
from monai.data import DataLoader, Dataset
from monai.data.image_reader import ITKReader
from monai.networks.nets import resnet18
import argparse

#%%
# Argument Parsing
parser = argparse.ArgumentParser(description="Convert bin to matrix, sort temporally and get jsons with additional info")
parser.add_argument("--path_input_patients", type=str, default="/your_path_in", help="Input path to folder containing patient subfolders")
parser.add_argument("--path_output_patients", type=str, default="/your_path_out", help="Output path where converted files are stored")
parser.add_argument("--path_classifier", type=str, default='/your_path_to_orientation_model/orientation_model/cnn_orientation_model_0.0016036431093155962.pth', help="Orientation classification neural network")
args = parser.parse_args()

#%%
# Variables and functions

plot=False  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 332  

def read_mha(path):
    """
    ​Given a path the function reads the image and returns header as well as the intensity values in a 3D array.
    :param path: String value that represents the path to the .mha file
    :return: image executable and array with intensity values
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    
    # get array with values from sitk image
    image_array = sitk.GetArrayFromImage(image)
    # note that SITK uses reverse indexing (eg h,w,d) compared to Numpy etc. (eg d,h,w)

    return image, image_array
 
    
def write_mha(path, image_array, spacing=None, origin=None):
    """
    Given a Dd array the function saves a .mha file at the requested path.
    :param path: String value that represents the path to the output .mha file
    :param image_array: 3D array with intensity values
    :param: spacing: optional list with image resolution, e.g. [1.3, 1.3, 5.0]
    :param: origin: optional list with image origin, e.g. [0.3, 0.3, 0.0]
    
    """   
    # get sitk image from array of values
    sitk_image = sitk.GetImageFromArray(image_array)
#    print('Size of image: ' + str(sitk_image.GetSize())) # e.g (1, 256, 256)

    # write header info into sitk image
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
        
    if origin is not None:
        sitk_image.SetOrigin(origin)
        
    # save image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(sitk_image)
    
    
# Get paths to mhas of one scan
def get_data(data_dir):
    data_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".mha"):
            data_list.append(
                {
                    "image": os.path.join(data_dir, file),
                    "path": os.path.join(data_dir, file),
                }
            )
    return data_list

def check_for_nans(data, label):
    
    if np.isnan(data).any():
        print(f"{label}: Min: {data.min()}, Max: {data.max()}, Has NaNs: {np.isnan(data).any()}")
        raise Exception('Attention! NaN found in data, check it.')
    
def check_for_misclassfied(path_to_output_folder_classified, path_log=None):
    "Check with pixel encoding whether some folders contains frames which are different from the remaining frames."
    
    dir_list = sorted(os.listdir(path_to_output_folder_classified))
    
    if len(dir_list) == 0:
        print('Attention: no files found in:', path_to_output_folder_classified)
        majority_class = None
        minority_files = None
    
    else:
        orientation_sums = []  
        file_names = [] 
        # Loop over each item in the dir_list
        for file_name in dir_list:
            if file_name.endswith('.mha'):
                path_input_file = os.path.join(path_to_output_folder_classified, file_name)
                
                # Read image 
                _, img_array = read_mha(path_input_file)
            
                # Get orientation encoding
                if os.path.basename(path_to_output_folder_classified) == 'axi':
                    orientation_values=img_array[0,:1,1:17]
                elif os.path.basename(path_to_output_folder_classified) == 'cor':
                    orientation_values=img_array[-1:,0,1:17]               
                elif os.path.basename(path_to_output_folder_classified) == 'sag':
                    orientation_values=img_array[-1:,-17:-1,0]
                else:
                    raise Exception ('Unknown orientation output folder:', os.path.basename(path_to_output_folder_classified))
            # plt.imshow(orientation_values)
            
                # Sum it up to generate a single value
                orientation_sum = np.sum(orientation_values)
                orientation_sums.append(orientation_sum)
                file_names.append(file_name)
        
        # Get most common value and check if there are different values
        # Count occurrences of each element
        counts = Counter(orientation_sums)
        majority_class, majority_count = counts.most_common(1)[0]

        if len(orientation_sums) == majority_count:
            print(f"All elements in the {os.path.basename(path_to_output_folder_classified)} folder are the same!")
            minority_files = None
        else:
            print(f"Found different patterns in the {os.path.basename(path_to_output_folder_classified)} folder!")
            # Identify the minority class(es)
            minority_files = [file_names[i] for i, value in enumerate(orientation_sums) if value != majority_class]
        majority_class = str(majority_class)  # for saving in json
        
    
    classification_log = {'majority_class': majority_class, 'minority_files': minority_files} 
    
    if path_log is not None:
        with open(path_log, "w") as f:
            json.dump(classification_log, f, indent=4)  
            
    return classification_log
    
#%% MAIN

# Loop over patients and scans
for patient_ID in sorted(os.listdir(args.path_input_patients))[19:]:  #  select subset e.g. [19:] from the F_20 onwards
    print('Current patient:', patient_ID)
    if os.path.isdir(os.path.join(args.path_input_patients, patient_ID)):
        for scan_ID in sorted(os.listdir(os.path.join(args.path_input_patients, patient_ID))):
            if os.path.isdir(os.path.join(args.path_input_patients, patient_ID, scan_ID)):
                path_to_input_folder = os.path.join(args.path_input_patients, patient_ID, scan_ID)
                print('Current scan:', path_to_input_folder)
                
                path_to_output_folder_classified_sag = os.path.join(args.path_output_patients, patient_ID, scan_ID, 'sag')
                path_to_output_folder_classified_cor = os.path.join(args.path_output_patients, patient_ID, scan_ID, 'cor')
                path_to_output_folder_classified_axi = os.path.join(args.path_output_patients, patient_ID, scan_ID, 'axi')
                os.makedirs(path_to_output_folder_classified_sag, exist_ok=True)
                os.makedirs(path_to_output_folder_classified_cor, exist_ok=True)
                os.makedirs(path_to_output_folder_classified_axi, exist_ok=True)
                
                # Copy over the jsons
                shutil.copy(os.path.join(path_to_input_folder, 'b-field-strength.json'), os.path.join(args.path_output_patients, patient_ID, scan_ID, 'b-field-strength.json'))
                shutil.copy(os.path.join(path_to_input_folder, 'scanned-region.json'), os.path.join(args.path_output_patients, patient_ID, scan_ID, 'scanned-region.json'))
                shutil.copy(os.path.join(path_to_input_folder, 'frame-rate.json'), os.path.join(args.path_output_patients, patient_ID, scan_ID, 'frame-rate.json'))

                # Get data for AI                
                infer_data = get_data(path_to_input_folder)

                infer_transforms = Compose(
                    [
                        LoadImageD(keys=["image"], reader=ITKReader()),
                        EnsureChannelFirstD(keys=["image"], channel_dim=-1),  # images have shape (270,270,1)
                        RotateD(keys=["image"], angle=np.pi/2, mode=("bilinear")),  # ensure it is correctly rotated
                        LambdaD(keys=["image"], func=lambda x: (check_for_nans(x, "After LoadImage"), x)[1]),
                        ScaleIntensityRangeD(
                            keys=["image"],
                            a_min=0,
                            a_max=1000,
                            b_min=0.0,
                            b_max=1.0,
                            clip=False,
                        ),
                        CenterSpatialCropD(
                            keys=["image"],
                            roi_size=[image_size, image_size]
                        ),
                        ToTensorD(keys=["image"]),
                    ]
                )
                
                infer_dataset = Dataset(data=infer_data, transform=infer_transforms)
                infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)  # batch size must be =1

                model = resnet18(spatial_dims=2, n_input_channels=1, num_classes=3).to(device)
                # Load weights of trained model
                model.load_state_dict(torch.load(args.path_classifier))
                model.eval()

                with torch.no_grad():
                    # Use tqdm to see progress
                    with tqdm.tqdm(infer_loader, unit="batch", desc=f"Orientation classification progress") as tepoch:
                        for infer_batch in tepoch:
                            input_image, input_path = infer_batch["image"].to(device), infer_batch["path"][0]
                            # input_image_spacing = infer_batch['image_meta_dict']['spacing'].tolist()[0]
                            
                            # check the image!
                            # print(input_image.shape) # (1,1,332,332)
                            # plt.imshow(input_image[0,0,:,:].detach().cpu().numpy(), cmap='gray') 
                            
                            # t0 = time.time()
                            outputs = model(input_image)
                            # print(f'Inference time: {(time.time() - t0)*1000} ms' )  # about 4 ms on gpu and 50 ms on cpu 
                            
                            if torch.argmax(outputs) == 0:
                                # print('Axial!')
                                path_output_file = os.path.join(path_to_output_folder_classified_axi, os.path.basename(input_path))
                                
                                # Load original image (no intensity scaling etc)
                                img, img_array = read_mha(input_path)

                                # Reshape and save image in axial orientation (h,w,1) 
                                img_array_reshaped = img_array[:,:,::-1].astype(np.float32) # note reversed indexing 
                                img_spacing_reshaped = [img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]]
                                img_origin_reshaped = [img.GetOrigin()[0], img.GetOrigin()[1], img.GetOrigin()[2]]
                                write_mha(path_output_file, img_array_reshaped, 
                                        spacing=img_spacing_reshaped,
                                        origin=img_origin_reshaped)
                                
                            elif torch.argmax(outputs) == 1:
                                # print('Coronal!')
                                path_output_file = os.path.join(path_to_output_folder_classified_cor, os.path.basename(input_path))
                                
                                # Load original image (no intensity scaling etc)
                                img, img_array = read_mha(input_path)

                                # Reshape and save image in sagittal orientation (h,1,w) 
                                img_array_reshaped = img_array[0,::-1,None,::-1].astype(np.float32) # note reversed indexing
                                img_spacing_reshaped = [img.GetSpacing()[0], img.GetSpacing()[2], img.GetSpacing()[1]]
                                img_origin_reshaped = [img.GetOrigin()[0], img.GetOrigin()[2], img.GetOrigin()[1]]
                                write_mha(path_output_file, img_array_reshaped, 
                                        spacing=img_spacing_reshaped,
                                        origin=img_origin_reshaped)

                            elif torch.argmax(outputs) == 2:
                                # print('Sagittal!')   
                                path_output_file = os.path.join(path_to_output_folder_classified_sag, os.path.basename(input_path))

                                # Load original image (no intensity scaling etc)
                                img, img_array = read_mha(input_path)

                                # Reshape and save image in sagittal orientation (1,h,w) 
                                img_array_reshaped = img_array[0,::-1,:,None].astype(np.float32) # note reversed indexing
                                img_spacing_reshaped = [img.GetSpacing()[2], img.GetSpacing()[0], img.GetSpacing()[1]]
                                img_origin_reshaped = [img.GetOrigin()[2], img.GetOrigin()[0], img.GetOrigin()[1]]
                                write_mha(path_output_file, img_array_reshaped, 
                                        spacing=img_spacing_reshaped,
                                        origin=img_origin_reshaped)
                            
                            else:
                                raise Exception('Unknown orientation!')        
                
                # Go through classfied folders and check using the pixel info in the 
                # top corner of the image whether all images in that folder have the same pattern
                classification_log_axi = check_for_misclassfied(path_to_output_folder_classified_axi,
                                                           path_log=os.path.join(os.path.dirname(path_to_output_folder_classified_axi), 'classification_log_axi.json'))
                classification_log_cor = check_for_misclassfied(path_to_output_folder_classified_cor,
                                                           path_log=os.path.join(os.path.dirname(path_to_output_folder_classified_axi), 'classification_log_cor.json'))
                classification_log_sag = check_for_misclassfied(path_to_output_folder_classified_sag,
                                                           path_log=os.path.join(os.path.dirname(path_to_output_folder_classified_axi), 'classification_log_sag.json'))
                print('...completed! Wrote log to file.')
 
