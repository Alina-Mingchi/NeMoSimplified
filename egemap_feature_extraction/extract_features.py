"""
Extract features from clean .wav files with OpenSmile
"""

import opensmile as sm
import librosa as lb
import pandas as pd
from pathlib import Path
import soundfile as sf
from tqdm import tqdm  # For progress bar

# User parameters
DATASET_NAME = 'pvqd_enh_chime' # 
FEATURE_SET = 'egemaps'  # Options: 'compare', 'egemaps'
PARTITION='tt' # tt, cv, tr
USE_FUNCTIONALS = False  # True = functionals, False = LLDs

INPUT_FOLDER = f'/idiap/temp/mhou/code/StoRM/{DATASET_NAME}/audio/{PARTITION}/clean/'  # Change to your folder containing WAV files
if USE_FUNCTIONALS:
    OUTPUT_CSV = f'/idiap/temp/mhou/code/feature_analysis/features_extracted/{PARTITION}/{DATASET_NAME}_features_{FEATURE_SET}.csv'  # Change to your desired output CSV path
else:
    OUTPUT_CSV = f'/idiap/temp/mhou/code/feature_analysis/features_extracted/{PARTITION}/{DATASET_NAME}_features_{FEATURE_SET}_LLDs.csv'  # Change to your desired output CSV path


# Initialize openSMILE extractor
if FEATURE_SET.lower() == 'compare':
    feature_set = sm.FeatureSet.ComParE_2016
elif FEATURE_SET.lower() == 'egemaps':
    feature_set = sm.FeatureSet.eGeMAPSv02
else:
    raise ValueError(f'Unknown feature set {FEATURE_SET}')

feature_level = sm.FeatureLevel.Functionals if USE_FUNCTIONALS else sm.FeatureLevel.LowLevelDescriptors

smile = sm.Smile(
    feature_set=feature_set,
    feature_level=feature_level,
)
print(f'Extractor ready → {FEATURE_SET} / ' + ('Functionals' if USE_FUNCTIONALS else 'LLDs'))

# Process all WAV files in the folder
input_folder = Path(INPUT_FOLDER)
wav_files = list(input_folder.rglob('*.wav'))  # Recursively find all WAV files
print(f'Found {len(wav_files)} WAV files in {INPUT_FOLDER}')

all_features = []

for i, wav_file in enumerate(tqdm(wav_files, desc="Processing files")):
    try:
        # Load audio
        audio, sr = lb.load(str(wav_file), sr=None, mono=True)
        
        # Extract features
        features_df = smile.process_signal(audio, sr)
        features_df['file'] = str(wav_file)  # Add filepath as a column
        
        # Append to the list
        all_features.append(features_df)
        
        # Print progress every 100 files
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(wav_files)} files...')
    except Exception as e:
        print(f'Error processing {wav_file}: {e}')

# Combine all features into a single DataFrame
if all_features:
    combined_df = pd.concat(all_features, ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f'Features saved to {OUTPUT_CSV}')
else:
    print('No features extracted.')