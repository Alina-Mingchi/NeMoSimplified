'''
Calculate metrics for each audio, write a csv
'''


from evaluation_metrics_new import evaluate_1utterance


import json
import os
import glob
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import re  # Regular expression module

SAMPLE_RATE = 16000

# Function to extract the SNR from the noisy file path
def extract_snr(noisy_filepath):
    # Regular expression to find snr= value in the filename
    match = re.search(r'snr=(-?\d+\.?\d*)', noisy_filepath)
    if match:
        return float(match.group(1))  # Extracted SNR value
    else:
        return None  # If snr= is not found, return None

# Get all JSON files in a folder:
# directory = '/idiap/temp/mhou/code/NeMo/'
# directory = '/idiap/temp/mhou/code/NeMo/enh_crowd_enh_chime_5corruptions_manifests'

# Method 1: Using os and listdir to get all .json files
# json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

# Method 2: Using glob to get all .json files (with full paths)
# json_files_full = glob.glob(os.path.join(directory, "*.json"))

# Method 3: give json directly
json_files_full = ['/idiap/temp/mhou/code/NeMoAug/enh_wsj0_lowf3utt_manifest.json']
# json_files_full = ['/idiap/temp/mhou/code/NeMoAug/enh_crowd_lowf3utt_manifest.json']

# Read the JSON manifest file
for manifest_path in json_files_full:
    data = []
    # Read the JSON Lines file and load each line as a separate JSON object
    with open(manifest_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # strip to remove any extra whitespace or newlines

    # data[0]
    # {'noisy_filepath': '/idiap/temp/mhou/code/StoRM/gita_enh_chime/audio_0/tr/noisy/AVPEPUDEA0001-Monologo-NR_634_snr=5.3.wav', 'speech_filepath': '/idiap/temp/mhou/code/StoRM/gita_enh_chime/audio_0/tr/clean/AVPEPUDEA0001-Monologo-NR_634_snr=5.3.wav', 'duration': 59.3714375, 'offset': 0}
    # data[0]['noisy_filepath']
    # '/idiap/temp/mhou/code/StoRM/gita_enh_chime/audio_0/tr/noisy/AVPEPUDEA0001-Monologo-NR_634_snr=5.3.wav'


    # Prepare the list to store the results
    results = []
    deltas = []

    # Iterate over the manifest data and extract the paths
    for i in range(len(data)):
        print(i)
        noisy_filepath =  data[i]['noisy_filepath']
        clean_filepath = data[i]['speech_filepath']
        processed_audio_filepath = data[i]['processed_audio_filepath']
        
        # Extract SNR value from the noisy_filepath
        snr_value = extract_snr(noisy_filepath)

        # Load audio files
        noisy, sr_noisy = librosa.load(noisy_filepath, sr=None)
        clean, sr_clean = librosa.load(clean_filepath, sr=None)
        enhanced, sr_enhanced = librosa.load(processed_audio_filepath, sr=None)


        # cut too long audios
        if len(clean) > 1600000:
            clean = clean[:1600000]
            noisy = noisy[:1600000]
            enhanced = enhanced[:1600000]
        
        # Calculate the metrics for each entry
        metrics_enh = evaluate_1utterance(clean, enhanced, SAMPLE_RATE)
        metrics_mix = evaluate_1utterance(clean, noisy, SAMPLE_RATE)
        metrics_delta = [a - b for a, b in zip(metrics_enh, metrics_mix)]
        # [ estoi_mean, pesq_mean, fwsnr_mean, sisdr_mean]
        # Store the results in the list
        results.append({
            'enhanced_filepath': processed_audio_filepath,
            'snr': snr_value,  # Add the SNR to the result
            'estoi': metrics_enh[0],        
            'pesqc2': metrics_enh[1], 
            'fwsnr': metrics_enh[2],        
            'sisdr': metrics_enh[3],        
            'mix estoi': metrics_mix[0],        
            'mix pesqc2': metrics_mix[1],  
            'mix fwsnr': metrics_mix[2],
            'mix sisdr': metrics_mix[3]
        })

        deltas.append({
            'enhanced_filepath': processed_audio_filepath,
            'snr': snr_value,  # Add the SNR to the result
            'estoi': metrics_delta[0],        
            'pesqc2': metrics_delta[1],        
            'fwsnr': metrics_delta[2],        
            'sisdr': metrics_delta[3]
        })
    # Define the column names explicitly
    columns = ['enhanced_filepath', 'snr', 'estoi', 'pesqc2', 'fwsnr', 'sisdr', 
            'mix estoi', 'mix pesqc2', 'mix fwsnr', 'mix sisdr']
            
    columns_deltas = ['enhanced_filepath', 'snr', 'estoi', 'pesqc2', 'fwsnr', 'sisdr']

    # Create a DataFrame with the column names
    df = pd.DataFrame(results, columns=columns)
    df_deltas = pd.DataFrame(deltas, columns=columns_deltas)

    file_name = os.path.basename(manifest_path)  # Extracts the file name 'enh_wsj0_sb_manifest.json'
    prefix = file_name.split('_manifest.json')[0]  # Splits the filename and takes the prefix part

    # Save the results to a CSV file
    output_csv_path = f'/idiap/temp/mhou/code/NeMoAug/all_dfs/{prefix}_PESQC2.csv'
    df.to_csv(output_csv_path, index=False)
    delta_output_csv_path = f'/idiap/temp/mhou/code/NeMoAug/all_deltas_dfs/{prefix}_PESQC2.csv'
    df_deltas.to_csv(delta_output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")


