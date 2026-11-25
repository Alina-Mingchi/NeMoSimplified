# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import sys
import os
root_dir = os.getcwd()
sys.path.append('/idiap/temp/mhou/code/NeMoBasic22kHz')
from enum import Enum

import lightning.pytorch as pl
import torch
torch.cuda.empty_cache()
from omegaconf import OmegaConf

from nemo.collections.audio.models.enhancement import (
    EncMaskDecAudioToAudioModel,
    FlowMatchingAudioToAudioModel,
    PredictiveAudioToAudioModel,
    SchroedingerBridgeAudioToAudioModel,
    ScoreBasedGenerativeAudioToAudioModel,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


class ModelType(str, Enum):
    """Enumeration with the available model types."""

    MaskBased = 'masking'
    Predictive = 'predictive'
    ScoreBased = 'score_based'
    SchroedingerBridge = 'schroedinger_bridge'
    FlowMatching = 'flow_matching'


def get_model_class(model_type: ModelType):
    """Get model class for a given model type."""
    if model_type == ModelType.MaskBased:
        return EncMaskDecAudioToAudioModel
    elif model_type == ModelType.Predictive:
        return PredictiveAudioToAudioModel
    elif model_type == ModelType.ScoreBased:
        return ScoreBasedGenerativeAudioToAudioModel
    elif model_type == ModelType.SchroedingerBridge:
        return SchroedingerBridgeAudioToAudioModel
    elif model_type == ModelType.FlowMatching:
        return FlowMatchingAudioToAudioModel
    else:
        raise ValueError(f'Unknown model type: {model_type}')


# Modify the callback to print only the first 10 elements of the specified weights
class CustomCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if name == "estimator.ncsnpp.input_pyramid.0.weight":
                logging.info(f"Epoch {trainer.current_epoch}: First 10 elements of {name}: {param.data.view(-1)[:10]}")


@hydra_runner(config_path="./conf", config_name="masking")
def main(cfg):
    # pl.seed_everything(42) ## could add for seed reproducibility
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg, resolve=True)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Get model class
    model_type = cfg.model.get('type')
    if model_type is None:
        model_type = ModelType.MaskBased
        logging.warning('model_type not found in config. Using default: %s', model_type)

    logging.info('Get class for model type: %s', model_type)
    model_class = get_model_class(model_type)

    logging.info('Instantiate model %s', model_class.__name__)
    model = model_class(cfg=cfg.model, trainer=trainer)

    # logging.info('Initialize the weights of the model from another model, if provided via config')
    # model.maybe_init_from_pretrained_checkpoint(cfg)

    # Add weight initialization logging
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"Initial weights of {name}: {param.data}")

    # # Add logging to print the weights of the specified layer
    # for name, param in model.named_parameters():
    #     if name == "estimator.ncsnpp.input_pyramid.0.weight":
    #         logging.info(f"Weights of {name}: {param.data}")

    # Train the model
    trainer.fit(model)

    # Run on test data, if available
    if hasattr(cfg.model, 'test_ds'):
        if trainer.is_global_zero:
            # Destroy the current process group and let the trainer initialize it again with a single device.
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            # Run test on a single device
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator)
            if model.prepare_test(trainer):
                trainer.test(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
