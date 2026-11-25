#!/bin/bash
#SBATCH --time 23:59:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition gpu
#SBATCH --gpus h100
#SBATCH --account

#SBATCH --job-name=train
#SBATCH --output=/code/slurm_logs/nemo-train-%j.out
#SBATCH --error=/code/slurm_logs/nemo-train-%j.err

#Setup
MAX_EPOCHS=1000 # max epochs
LEARNING_RATE=0.0001
BATCH=1
CONFIG_NAME='predictive'
DATASET='wsj0_enh_chime'


conda activate nemo


echo 'Train SE with Nemo for' $DATASET

cd /code/NeMoSimplified/
python /code/NeMoSimplified/examples/audio/audio_to_audio_train_scratch.py \
        --config-path='/code/NeMoSimplified/examples/audio/conf' --config-name ${CONFIG_NAME} \
        model.train_ds.manifest_filepath='/code/'${DATASET}'/dataset_train.json' \
        model.validation_ds.manifest_filepath='/code/'${DATASET}'/dataset_val.json' \
        model.train_ds.input_key='noisy_filepath'\
        model.train_ds.target_key='speech_filepath'\
        model.train_ds.batch_size=${BATCH}\
        model.validation_ds.input_key='noisy_filepath'\
        model.validation_ds.target_key='speech_filepath'\
        model.validation_ds.batch_size=${BATCH}\
        model.optim.lr=${LEARNING_RATE}\
        trainer.devices=1 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=${MAX_EPOCHS}

