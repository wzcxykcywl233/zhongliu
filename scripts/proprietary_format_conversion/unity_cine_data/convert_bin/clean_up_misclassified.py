import os
import json
import shutil

# Input and output dirs
base_dir = '/your_path/by_orientation'
output_dir = '/your_path/by_orientation_cleaned_up'

# Lists to hold the scan paths for each category.
all_work = []      # Category 1: All three logs are "successful"
partial_fail = []  # Category 2: Some logs work and some fail
total_fail = []    # Category 3: All three logs fail

# Define the expected log file names for the three orientations.
log_files = {
    'axi': 'classification_log_axi.json',
    'cor': 'classification_log_cor.json',
    'sag': 'classification_log_sag.json'
}

def classification_works(data):
    """
    Determines if the classification for a given orientation worked.
    It works if 'majority_class' is not None and 'minority_files' is None.
    """
    return data.get("majority_class") is not None and data.get("minority_files") is None

# Iterate over each patient folder.
for patient in sorted(os.listdir(base_dir))[:]:
    patient_path = os.path.join(base_dir, patient)
    if not os.path.isdir(patient_path):
        continue
    # Iterate over each scan folder inside the patient folder.
    for scan in sorted(os.listdir(patient_path)):
        scan_path = os.path.join(patient_path, scan)
        if not os.path.isdir(scan_path):
            continue

        working_count = 0  # Count how many orientations worked.
        # Loop through each orientation log.
        for orientation, filename in log_files.items():
            file_path = os.path.join(scan_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    # If there's an error reading/parsing the file, mark as not working.
                    data = {}
            else:
                # Missing file is considered a failure.
                data = {}
            
            if classification_works(data):
                working_count += 1

        # Categorize the scan based on the number of orientations that worked.
        if working_count == 3:
            all_work.append(scan_path)
        elif working_count == 0:
            total_fail.append(scan_path)
        else:
            partial_fail.append(scan_path)

# Print out statistics.
total_scans = len(all_work) + len(partial_fail) + len(total_fail)
print("Total scans processed:", total_scans)
print("Classification worked for all three orientations:", len(all_work))
print("Classification partially worked (only 1 or 2 orientations):", len(partial_fail))
print("Classification failed for all three orientations:", len(total_fail))

# Save the log with the categorized paths.
log_filename = 'classification_summary_clean_up.log'
with open(os.path.join(output_dir, log_filename), 'w') as logfile:
    logfile.write("Scans where classification worked for all three orientations:\n")
    logfile.write("\n".join(all_work) + "\n\n")
    logfile.write("Scans where classification partially worked (1 or 2 orientations):\n")
    logfile.write("\n".join(partial_fail) + "\n\n")
    logfile.write("Scans where classification failed for all three orientations:\n")
    logfile.write("\n".join(total_fail) + "\n")

print(f"\nLog file saved as '{log_filename}'")

# Copy the scans from category one (all_work) and then of category three (total_fail) to output_dir 
# --> at was observed that these are the ones for which the classification worked. In the observed cases, category three failed completely just because 
# the scan was paused and the resumed which leads to a different pattern in the top left part of the image...
for scan_path in all_work + total_fail:
    # Compute the relative path from base_dir so that the patient/scan structure is preserved.
    rel_path = os.path.relpath(scan_path, base_dir)
    dest_path = os.path.join(output_dir, rel_path)
    # Ensure the destination parent directory exists.
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Copy the entire scan folder. If using Python 3.8+, dirs_exist_ok=True avoids errors if dest exists.
    shutil.copytree(scan_path, dest_path, dirs_exist_ok=True)
    print(f"Copied {scan_path} to {dest_path}")

print("\nCopy operation complete.")
