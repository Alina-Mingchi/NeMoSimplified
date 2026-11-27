"""
This is a customized library for evaluation of speech enhancement
Currently including  stoi, estoi, pesq, fwsnr, sisdr --> optional: replace pesq with pesqc2
It will return mean and 95% CI (evaluate)
return value (evaluate_1utterance)
"""

import numpy as np
from pystoi.stoi import stoi
import pysepm
import statistics
import scipy.stats as stats

import torch
import torchaudio.functional as F
from torchmetrics.audio import *
import pesqc2 

PESQC2 = True


def evaluate(CleanSig, EnhancedSig, fs):
    sstoi = []
    estoi = []
    pesq = []
    fwsnr = []
    sisdr = []
    si_SDR = ScaleInvariantSignalDistortionRatio()
    
    if type(CleanSig) == list:
        for i in range(len(CleanSig)):
            if i % 100 == 0:
                print(f"processing {i} / {len(CleanSig)}")
            sstoi.append(stoi(CleanSig[i],EnhancedSig[i],fs))
            estoi.append(stoi(CleanSig[i],EnhancedSig[i],fs,extended=True))
            if PESQC2:
                pesq.append(pesqc2.pesq(fs, CleanSig[i], EnhancedSig[i], 'wb'))
                # print('Using PESQC2')
            else:
                pesq.append(pysepm.pesq(CleanSig[i], EnhancedSig[i], fs)[1])
                # print('Using pysepm PESQ')
            fwsnr.append(pysepm.fwSNRseg(CleanSig[i],EnhancedSig[i],fs))
            sisdr.append(si_SDR(torch.Tensor(EnhancedSig[i]),torch.Tensor(CleanSig[i])))
        
        stoi_std = statistics.stdev(sstoi)
        estoi_std = statistics.stdev(estoi)  
        pesq_std = statistics.stdev(pesq)
        fwsnr_std = statistics.stdev(fwsnr)
        sisdr_std = torch.std(torch.stack(sisdr)).item()

    else:
        sstoi.append(stoi(CleanSig,EnhancedSig,fs))
        estoi.append(stoi(CleanSig,EnhancedSig,fs,extended=True))
        pesq.append(pysepm.pesq(CleanSig,EnhancedSig,fs)[1])
        fwsnr.append(pysepm.fwSNRseg(CleanSig,EnhancedSig,fs))
        sisdr.append(si_SDR(torch.Tensor(EnhancedSig),torch.Tensor(CleanSig)))
        stoi_std = 0
        estoi_std = 0
        pesq_std = 0
        fwsnr_std = 0
        sisdr_std = 0


    stoi_mean = statistics.mean(sstoi)
    estoi_mean = statistics.mean(estoi)
    pesq_mean = statistics.mean(pesq)
    fwsnr_mean = statistics.mean(fwsnr)
    sisdr_mean = torch.mean(torch.stack(sisdr)).item()

    stoi_med = statistics.median(sstoi)
    estoi_med = statistics.median(estoi)
    pesq_med = statistics.median(pesq)
    fwsnr_med = statistics.median(fwsnr)
    sisdr_med = torch.median(torch.stack(sisdr)).item()


    # value_list = [stoi_mean, stoi_std, estoi_mean, estoi_std, pesq_mean, pesq_std, fwsnr_mean, fwsnr_std, sisdr_mean, sisdr_std]
    # return list(map(lambda x: float(format(x, '.3f')), value_list))
    
    #############################################
    ### Calculate confidence interval
    #############################################
    # Calculate the standard error of the mean (SEM)
    n = len(sstoi)
    sem_stoi = stoi_std / np.sqrt(n)
    # print(f'Standard Error of the Mean (SEM): {sem_si_sdr}')

    # Find the critical value for a 95% confidence interval
    confidence_level = 0.95
    degrees_of_freedom = n - 1
    critical_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    # print(f'Critical Value (t*): {critical_value}')

    # Calculate the margin of error
    stoi_margin_of_error = critical_value * sem_stoi
    # print(f'Margin of Error: {margin_of_error}')

    # Repeat for ESTOI
    sem_estoi = estoi_std / np.sqrt(n)
    estoi_margin_of_error = critical_value * sem_estoi

    # Repeat for PESQ
    sem_pesq = pesq_std / np.sqrt(n)
    pesq_margin_of_error = critical_value * sem_pesq

    # Repeat for fwSNR
    sem_fwsnr = fwsnr_std / np.sqrt(n)
    fwsnr_margin_of_error = critical_value * sem_fwsnr

    # Repeat for SISDR
    sem_si_sdr = sisdr_std / np.sqrt(n)
    sisdr_margin_of_error = critical_value * sem_si_sdr

    print('Median, Mean, CI for STOI, ESTOI, PESQ, PESQC2,fwSNR, SI-SDR')
    value_list = [stoi_med, stoi_mean, stoi_margin_of_error, 
                  estoi_med, estoi_mean, estoi_margin_of_error, 
                  pesq_med, pesq_mean, pesq_margin_of_error, 
                  fwsnr_med, fwsnr_mean, fwsnr_margin_of_error,
                  sisdr_med, sisdr_mean, sisdr_margin_of_error]

    return list(map(lambda x: float(format(x, '.3f')), value_list))


def evaluate_1utterance(CleanSig, EnhancedSig, fs):
    estoi = []
    pesq = []
    fwsnr = []
    sisdr = []
    si_SDR = ScaleInvariantSignalDistortionRatio()
    
    estoi.append(stoi(CleanSig,EnhancedSig,fs,extended=True))
    if PESQC2:
        pesq.append(pesqc2.pesq(fs, CleanSig, EnhancedSig, 'wb'))
        # print('PESQC2')
    else:
        pesq.append(pysepm.pesq(CleanSig,EnhancedSig,fs)[1])
    fwsnr.append(pysepm.fwSNRseg(CleanSig,EnhancedSig,fs))
    sisdr.append(si_SDR(torch.Tensor(EnhancedSig),torch.Tensor(CleanSig)))


    estoi_mean = statistics.mean(estoi)
    pesq_mean = statistics.mean(pesq)
    fwsnr_mean = statistics.mean(fwsnr)
    sisdr_mean = torch.mean(torch.stack(sisdr)).item()


    # print('1 Utterance metrics STOI, ESTOI, PESQ, fwSNR, SI-SDR')
    value_list = [ estoi_mean, pesq_mean, fwsnr_mean, sisdr_mean]

    return list(map(lambda x: float(format(x, '.4f')), value_list))






















