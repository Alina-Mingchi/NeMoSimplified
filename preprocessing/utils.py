#!/usr/env/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import os
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve, stft
import glob
import itertools, operator
from librosa import resample
import librosa

def obtain_noise_file(noise_dir, i_sample, channels, dataset, sample_rate, len_speech):

    nb_samples = len(os.listdir(noise_dir))

    if dataset == "wham":
        noise_sample, noise_sr = sf.read(os.path.join(noise_dir, os.listdir(noise_dir)[i_sample%nb_samples]))

        if noise_sr != sample_rate: #Resample
            noise_sample = resample(noise_sample, orig_sr=noise_sr, target_sr=sample_rate)
        if channels == 1:
            noise_sample = noise_sample[:, 0]
            
    elif dataset == "chime":
        noise_types = ["CAF", "PED", "STR", "BUS"]
        noise_type = noise_types[np.random.randint(len(noise_types))]
        noise_candidates = glob.glob(os.path.join(noise_dir, f"*_{noise_type}.CH1.wav"))
        noise_sample_basename = noise_candidates[np.random.randint(len(noise_candidates))][: -8]
        noise_sample_ch1, noise_sr = sf.read(noise_sample_basename + ".CH1.wav")

        if noise_sr != sample_rate: #Resample
            noise_sample_ch1 = resample(noise_sample_ch1, orig_sr=noise_sr, target_sr=sample_rate)
            
        start = np.random.randint(noise_sample_ch1.shape[-1]-len_speech)
        noise_sample = np.stack([ sf.read(noise_sample_basename + f".CH{i_ch+1}.wav")[0].squeeze()[start: start+len_speech]
            for i_ch in range(channels) ])

        if noise_sr != sample_rate: #Resample
            noise_sample_resampled = np.stack([ resample(noise_sample_ch, orig_sr=noise_sr, target_sr=sample_rate) for noise_sample_ch in noise_sample ])
            noise_sample = noise_sample_resampled

    elif dataset == "chime2type":
        noise_types = ["CAF", "PED"]
        noise_type = noise_types[np.random.randint(len(noise_types))]
        noise_candidates = glob.glob(os.path.join(noise_dir, f"*_{noise_type}.CH1.wav"))
        noise_sample_basename = noise_candidates[np.random.randint(len(noise_candidates))][: -8]
        noise_sample_ch1, noise_sr = sf.read(noise_sample_basename + ".CH1.wav")

        if noise_sr != sample_rate: #Resample
            noise_sample_ch1 = resample(noise_sample_ch1, orig_sr=noise_sr, target_sr=sample_rate)
            
        start = np.random.randint(noise_sample_ch1.shape[-1]-len_speech)
        noise_sample = np.stack([ sf.read(noise_sample_basename + f".CH{i_ch+1}.wav")[0].squeeze()[start: start+len_speech]
            for i_ch in range(channels) ])

        if noise_sr != sample_rate: #Resample
            noise_sample_resampled = np.stack([ resample(noise_sample_ch, orig_sr=noise_sr, target_sr=sample_rate) for noise_sample_ch in noise_sample ])
            noise_sample = noise_sample_resampled

    elif dataset == "qut": ### THIS IS NEW to be most efficient, only load needed length 
        # info = sf.info(os.path.join(noise_dir, f"{noise_type}-1.wav")) #actually 40hours long, and 44.1kHz
        noise_types = ["CAFE-CAFE", "CAR-WINDOWNB", "HOME-KITCHEN", "STREET-CITY"]
        noise_type = np.random.choice(noise_types)
        noise_path = os.path.join(noise_dir, f"{noise_type}-1.wav")

        # Length we need (speech length in seconds)
        seg_duration = len_speech / sample_rate

        # Read metadata only (fast, no audio loaded)
        with sf.SoundFile(noise_path) as f:
            total_duration = f.frames / f.samplerate

        # Pick random offset (safe even for very long files)
        max_offset = max(0.0, total_duration - seg_duration)
        offset = np.random.uniform(0.0, max_offset)

        # Load only the needed segment, already resampled
        noise_sample, noise_sr = librosa.load(
            noise_path,
            sr=sample_rate,
            mono=True,
            offset=offset,
            duration=seg_duration
        )

        # Safety check (rare edge cases)
        if len(noise_sample) < len_speech:
            noise_sample = np.pad(
                noise_sample,
                (0, len_speech - len(noise_sample)),
                mode="wrap"
            )

    return noise_sample, noise_sr


def obtain_noise_file_output_with_type(noise_dir, i_sample, channels, dataset, sample_rate, len_speech):

    nb_samples = len(os.listdir(noise_dir))

    if dataset == "wham":
        noise_sample, noise_sr = sf.read(os.path.join(noise_dir, os.listdir(noise_dir)[i_sample%nb_samples]))

        if noise_sr != sample_rate: #Resample
            noise_sample = resample(noise_sample, orig_sr=noise_sr, target_sr=sample_rate)
        if channels == 1:
            noise_sample = noise_sample[:, 0]
            
    elif dataset == "chime":
        noise_types = ["CAF", "PED", "STR", "BUS"]
        noise_type = noise_types[np.random.randint(len(noise_types))]
        noise_candidates = glob.glob(os.path.join(noise_dir, f"*_{noise_type}.CH1.wav"))
        noise_sample_basename = noise_candidates[np.random.randint(len(noise_candidates))][: -8]
        noise_sample_ch1, noise_sr = sf.read(noise_sample_basename + ".CH1.wav")

        if noise_sr != sample_rate: #Resample
            noise_sample_ch1 = resample(noise_sample_ch1, orig_sr=noise_sr, target_sr=sample_rate)
            
        start = np.random.randint(noise_sample_ch1.shape[-1]-len_speech)
        noise_sample = np.stack([ sf.read(noise_sample_basename + f".CH{i_ch+1}.wav")[0].squeeze()[start: start+len_speech]
            for i_ch in range(channels) ])

        if noise_sr != sample_rate: #Resample
            noise_sample_resampled = np.stack([ resample(noise_sample_ch, orig_sr=noise_sr, target_sr=sample_rate) for noise_sample_ch in noise_sample ])
            noise_sample = noise_sample_resampled

    elif dataset == "chime2type":
        noise_types = ["PED"]
        noise_type = noise_types[np.random.randint(len(noise_types))]
        noise_candidates = glob.glob(os.path.join(noise_dir, f"*_{noise_type}.CH1.wav"))
        noise_sample_basename = noise_candidates[np.random.randint(len(noise_candidates))][: -8]
        noise_sample_ch1, noise_sr = sf.read(noise_sample_basename + ".CH1.wav")

        if noise_sr != sample_rate: #Resample
            noise_sample_ch1 = resample(noise_sample_ch1, orig_sr=noise_sr, target_sr=sample_rate)
            
        start = np.random.randint(noise_sample_ch1.shape[-1]-len_speech)
        noise_sample = np.stack([ sf.read(noise_sample_basename + f".CH{i_ch+1}.wav")[0].squeeze()[start: start+len_speech]
            for i_ch in range(channels) ])

        if noise_sr != sample_rate: #Resample
            noise_sample_resampled = np.stack([ resample(noise_sample_ch, orig_sr=noise_sr, target_sr=sample_rate) for noise_sample_ch in noise_sample ])
            noise_sample = noise_sample_resampled


    elif dataset == "qut": ### THIS IS NEW to be most efficient, only load needed length 
        # info = sf.info(os.path.join(noise_dir, f"{noise_type}-1.wav")) #actually 40hours long, and 44.1kHz
        noise_types = ["CAFE-CAFE", "CAR-WINDOWNB", "HOME-KITCHEN", "STREET-CITY"]
        noise_type = np.random.choice(noise_types)
        noise_path = os.path.join(noise_dir, f"{noise_type}-1.wav")

        # Length we need (speech length in seconds)
        seg_duration = len_speech / sample_rate

        # Read metadata only (fast, no audio loaded)
        with sf.SoundFile(noise_path) as f:
            total_duration = f.frames / f.samplerate

        # Pick random offset (safe even for very long files)
        max_offset = max(0.0, total_duration - seg_duration)
        offset = np.random.uniform(0.0, max_offset)

        # Load only the needed segment, already resampled
        noise_sample, noise_sr = librosa.load(
            noise_path,
            sr=sample_rate,
            mono=True,
            offset=offset,
            duration=seg_duration
        )

        # Safety check (rare edge cases)
        if len(noise_sample) < len_speech:
            noise_sample = np.pad(
                noise_sample,
                (0, len_speech - len(noise_sample)),
                mode="wrap"
            )

    return noise_sample, noise_sr, noise_type
