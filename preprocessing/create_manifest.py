"""
Create manifest that could be used for NVIDIA NeMo training
"""


import os
import json
import wave

def get_duration(filepath):
    with wave.open(filepath, 'rb') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def generate_manifest(clean_dir, noisy_dir, output_json):
    manifest = []

    # List all files in the clean and noisy directories
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
    print('Clean files number: ' + str(len(clean_files)))
    print('Noisy files number: ' + str(len(noisy_files)))

    # Ensure the number of clean and noisy files are the same
    if len(clean_files) != len(noisy_files):
        raise ValueError("Number of clean files and noisy files do not match!")

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_filepath = os.path.join(clean_dir, clean_file)
        noisy_filepath = os.path.join(noisy_dir, noisy_file)

        # Get the duration of the clean and noisy files
        duration = get_duration(clean_filepath)
        offset = 0  # Assuming offset is 0 for simplicity

        # Create the entry for the manifest
        entry = {
            "noisy_filepath": noisy_filepath,
            "speech_filepath": clean_filepath,
            "duration": duration,
            "offset": offset
        }

        # Append the entry to the manifest list
        manifest.append(entry)

    # Write the manifest to the output JSON file
    with open(output_json, 'w') as f:
        for entry in manifest:
            json.dump(entry, f)
            f.write("\n")


ALL_AS_TEST = True # Set to True if the whole dataset is used as test. 
# e.g. when you want to take a pretrained model and investigate the performance on a new dataset

if __name__ == "__main__":
    #########################################
    ### Data in 1 folder 
    #########################################
    if ALL_AS_TEST == True:
        DATASET = "pvqd_enh_chime_all_as_test"
        dic_split = {"tt": "test"}
    
    DATASET = "pvqd_enh_chime"
    dic_split = {"tt": "test", "cv": "val", "tr": "train"}

    for split in dic_split:
        clean_directory = f"/idiap/temp/mhou/code/StoRM/{DATASET}/audio/{split}/clean/"
        noisy_directory = f"/idiap/temp/mhou/code/StoRM/{DATASET}/audio/{split}/noisy/"
        output_json_file = f"/idiap/temp/mhou/code/StoRM/{DATASET}/dataset_{dic_split[split]}.json"

        generate_manifest(clean_directory, noisy_directory, output_json_file)
        print(f"Manifest has been written to {output_json_file}")

    #########################################
    ### 10 fold stratified cross validation 
    #########################################

    # DATASET = "gita_enh_chime_4corruptions"
    # dic_split = {"tr": "train"}
    # for fold_idx in range(10):
    #     for split in dic_split:
    #         clean_directory = f"/idiap/temp/mhou/code/StoRM/{DATASET}/audio_{fold_idx}/{split}/clean/"
    #         noisy_directory = f"/idiap/temp/mhou/code/StoRM/{DATASET}/audio_{fold_idx}/{split}/noisy/"
    #         output_json_file = f"/idiap/temp/mhou/code/StoRM/{DATASET}/dataset_{dic_split[split]}_{fold_idx}.json"

    #         generate_manifest(clean_directory, noisy_directory, output_json_file)
    #         print(f"Manifest has been written to {output_json_file}")
