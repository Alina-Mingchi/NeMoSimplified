#!/bin/bash
#SBATCH --time 00:20:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition gpu
#SBATCH --gpus 1


#SBATCH --job-name=enh
#SBATCH --output=/code/slurm_logs/nemo-enh-%j.out
#SBATCH --error=/code/slurm_logs/nemo-enh-%j.err

#Setup
wsj0_model_pred='/code/NeMoSimplified/predictive.nemo'
MODEL_PATH=$wsj0_model_pred
DATASET='wsj0_enh_chime'


conda activate nemo

echo 'Enhance SE with Nemo for' $DATASET


python /code/NeMoSimplified/examples/audio/audio_to_audio_eval.py \
    model_path=${MODEL_PATH} \
    dataset_manifest='/code/StoRM/'${DATASET}'/dataset_test.json'\
    output_dir='/code/NeMoSimplified/'${DATASET}'_pred'\
    target_key='speech_filepath'\
    metrics=[pesq,estoi,sisdr]\
    batch_size=1 \
    amp=True    