# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
from typing import Dict, Optional, Sequence

import einops
import einops.layers.torch
import torch
import torch.nn.functional as F

from nemo.collections.common.parts.utils import activation_registry, mask_sequence_tensor
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import FloatType, LengthsType, NeuralType, SpectrogramType, VoidType
from nemo.utils import logging


class SpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = NoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params)

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length


class NoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        **_,
    ):
        # Network topology is a flavor of UNet, example chart for num_resolutions=4
        #
        # 1: Image  → Image/2  → Image/4  → Image/8
        #       ↓        ↓          ↓          ↓
        # 2: Hidden → Hidden/2 → Hidden/4 → Hidden/8
        #       ↓        ↓          ↓          ↓
        # 3: Hidden ← Hidden/2 ← Hidden/4 ← Hidden/8
        #       ↓        ↓          ↓          ↓
        # 4: Image  ← Image/2  ← Image/4  ← Image/8

        # Horizontal arrows in (1) are downsampling
        # Vertical arrows from (1) to (2) are channel upconversions
        #
        # Horizontal arrows in (2) are blocks with downsampling where necessary
        # Horizontal arrows in (3) are blocks with upsampling where necessary
        #
        # Vertical arrows from (1) to (2) are downsampling and channel upconversioins
        # Vertical arrows from (2) to (3) are sums connections (also with / sqrt(2))
        # Vertical arrows from (3) to (4) are channel downconversions
        # Horizontal arrows in (4) are upsampling and addition
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))

        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class NTTSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = NTTNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params)

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###NTT
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class NTTNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):
        # Network topology is a flavor of UNet, example chart for num_resolutions=4
        #
        # 1: Image  → Image/2  → Image/4  → Image/8
        #       ↓        ↓          ↓          ↓
        # 2: Hidden → Hidden/2 → Hidden/4 → Hidden/8
        #       ↓        ↓          ↓          ↓
        # 3: Hidden ← Hidden/2 ← Hidden/4 ← Hidden/8
        #       ↓        ↓          ↓          ↓
        # 4: Image  ← Image/2  ← Image/4  ← Image/8

        # Horizontal arrows in (1) are downsampling
        # Vertical arrows from (1) to (2) are channel upconversions
        #
        # Horizontal arrows in (2) are blocks with downsampling where necessary
        # Horizontal arrows in (3) are blocks with upsampling where necessary
        #
        # Vertical arrows from (1) to (2) are downsampling and channel upconversioins
        # Vertical arrows from (2) to (3) are sums connections (also with / sqrt(2))
        # Vertical arrows from (3) to (4) are channel downconversions
        # Horizontal arrows in (4) are upsampling and addition
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_hidden_channels = 128 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        # self.embedding_dim = embedding_dim ###DEBUG
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_scale = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.first_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.first_hidden_channels, self.first_hidden_channels),
        ) ###NTT

        torch.nn.init.xavier_uniform_(self.embedding_to_scale[-1].weight) ###NTT
        torch.nn.init.zeros_(self.embedding_to_scale[-1].bias) ###NTT

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in NTT used after 1st Resnet block
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        # for resolution_num, image in enumerate(pyramid): ###DEBUG
        #     print(f"resolution_num: {resolution_num}") ###DEBUG printout-> 0,1,2,3 -> each one has a different hidden size after 1st resblock
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            resblock_idx = 0  # before the loop (define once) ###NTT
            # print(f"Total number of resblock: {self.num_res_blocks}") ###DEBUG
            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)###NTT
                # print("hidden shape:", hidden.shape) ###DEBUG
                # print("hidden type:", hidden.dtype) ###DEBUG

                # multiplicative adaptation AFTER FIRST RESNET BLOCK ###NTT
                if resblock_idx == 0 and resolution_num == 0 and spkr_embedding is not None:
                    spkr_embedding = spkr_embedding.to(hidden.dtype)
                    # print("spkr_embedding shape:", spkr_embedding.shape) ###DEBUG
                    # print("spkr_embedding type:", spkr_embedding.dtype) ###DEBUG
                    scale = self.embedding_to_scale(spkr_embedding)  # (B, C)
                    scale = scale[:, :, None, None]             # (B, C, 1, 1)
                    # print("scale shape:", scale.shape) ###DEBUG
                    # print("scale type:", scale.dtype) ###DEBUG
                    # hidden = hidden * torch.tanh(1.0 + scale) ### Model saved as 20260110
                    hidden = hidden *  scale
                    # hidden = hidden + scale

                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

                resblock_idx += 1
                # print(f"resblock_idx: {resblock_idx}") ###DEBUG

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class NTTAddSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = NTTAddNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params)

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###NTT
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class NTTAddNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):
        # Network topology is a flavor of UNet, example chart for num_resolutions=4
        #
        # 1: Image  → Image/2  → Image/4  → Image/8
        #       ↓        ↓          ↓          ↓
        # 2: Hidden → Hidden/2 → Hidden/4 → Hidden/8
        #       ↓        ↓          ↓          ↓
        # 3: Hidden ← Hidden/2 ← Hidden/4 ← Hidden/8
        #       ↓        ↓          ↓          ↓
        # 4: Image  ← Image/2  ← Image/4  ← Image/8

        # Horizontal arrows in (1) are downsampling
        # Vertical arrows from (1) to (2) are channel upconversions
        #
        # Horizontal arrows in (2) are blocks with downsampling where necessary
        # Horizontal arrows in (3) are blocks with upsampling where necessary
        #
        # Vertical arrows from (1) to (2) are downsampling and channel upconversioins
        # Vertical arrows from (2) to (3) are sums connections (also with / sqrt(2))
        # Vertical arrows from (3) to (4) are channel downconversions
        # Horizontal arrows in (4) are upsampling and addition
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_hidden_channels = 128 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        # self.embedding_dim = embedding_dim ###DEBUG
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_scale = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.first_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.first_hidden_channels, self.first_hidden_channels),
        ) ###NTT

        torch.nn.init.xavier_uniform_(self.embedding_to_scale[-1].weight) ###NTT
        torch.nn.init.zeros_(self.embedding_to_scale[-1].bias) ###NTT

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in NTT used after 1st Resnet block
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        # for resolution_num, image in enumerate(pyramid): ###DEBUG
        #     print(f"resolution_num: {resolution_num}") ###DEBUG printout-> 0,1,2,3 -> each one has a different hidden size after 1st resblock
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            resblock_idx = 0  # before the loop (define once) ###NTT
            # print(f"Total number of resblock: {self.num_res_blocks}") ###DEBUG
            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)###NTT
                # print("hidden shape:", hidden.shape) ###DEBUG
                # print("hidden type:", hidden.dtype) ###DEBUG

                # multiplicative adaptation AFTER FIRST RESNET BLOCK ###NTT
                if resblock_idx == 0 and resolution_num == 0 and spkr_embedding is not None:
                    spkr_embedding = spkr_embedding.to(hidden.dtype)
                    # print("spkr_embedding shape:", spkr_embedding.shape) ###DEBUG
                    # print("spkr_embedding type:", spkr_embedding.dtype) ###DEBUG
                    scale = self.embedding_to_scale(spkr_embedding)  # (B, C)
                    scale = scale[:, :, None, None]             # (B, C, 1, 1)
                    # print("scale shape:", scale.shape) ###DEBUG
                    # print("scale type:", scale.dtype) ###DEBUG
                    # hidden = hidden * torch.tanh(1.0 + scale) ### Model saved as 20260110
                    # hidden = hidden *  scale
                    hidden = hidden + scale

                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

                resblock_idx += 1
                # print(f"resblock_idx: {resblock_idx}") ###DEBUG

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class NTTFiLMSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = NTTFiLMNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params)

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###NTT
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class NTTFiLMNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):
        # Network topology is a flavor of UNet, example chart for num_resolutions=4
        #
        # 1: Image  → Image/2  → Image/4  → Image/8
        #       ↓        ↓          ↓          ↓
        # 2: Hidden → Hidden/2 → Hidden/4 → Hidden/8
        #       ↓        ↓          ↓          ↓
        # 3: Hidden ← Hidden/2 ← Hidden/4 ← Hidden/8
        #       ↓        ↓          ↓          ↓
        # 4: Image  ← Image/2  ← Image/4  ← Image/8

        # Horizontal arrows in (1) are downsampling
        # Vertical arrows from (1) to (2) are channel upconversions
        #
        # Horizontal arrows in (2) are blocks with downsampling where necessary
        # Horizontal arrows in (3) are blocks with upsampling where necessary
        #
        # Vertical arrows from (1) to (2) are downsampling and channel upconversioins
        # Vertical arrows from (2) to (3) are sums connections (also with / sqrt(2))
        # Vertical arrows from (3) to (4) are channel downconversions
        # Horizontal arrows in (4) are upsampling and addition
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_hidden_channels = 128 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_film = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.first_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.first_hidden_channels, 2 * self.first_hidden_channels),
        ) ###NTT

        torch.nn.init.xavier_uniform_(self.embedding_to_film[-1].weight) ###NTT
        torch.nn.init.zeros_(self.embedding_to_film[-1].bias) ###NTT

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in NTT used after 1st Resnet block
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        # for resolution_num, image in enumerate(pyramid): ###DEBUG
        #     print(f"resolution_num: {resolution_num}") ###DEBUG printout-> 0,1,2,3 -> each one has a different hidden size after 1st resblock
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            resblock_idx = 0  # before the loop (define once) ###NTT
            # print(f"Total number of resblock: {self.num_res_blocks}") ###DEBUG
            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)###NTT
                # print("hidden shape:", hidden.shape) ###DEBUG
                # print("hidden type:", hidden.dtype) ###DEBUG

                # multiplicative adaptation AFTER FIRST RESNET BLOCK ###NTT
                if resblock_idx == 0 and resolution_num == 0 and spkr_embedding is not None:
                    spkr_embedding = spkr_embedding.to(hidden.dtype)
                    # print("spkr_embedding shape:", spkr_embedding.shape) ###DEBUG
                    # print("spkr_embedding type:", spkr_embedding.dtype) ###DEBUG
                    scale, shift  = self.embedding_to_film(spkr_embedding).chunk(2, dim=1)  # (B, C)
                    # print("scale type:", scale.dtype) ###DEBUG
                    # print("scale shape:", scale.shape, "scale(B, :4) ", scale[:, :4]) ###DEBUG
                    # print("shift shape:", shift.shape, "shift(B, :4) ", shift[:, :4]) ###DEBUG
                    hidden = hidden * scale[:, :, None, None] + shift[:, :, None, None]


                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

                resblock_idx += 1
                # print(f"resblock_idx: {resblock_idx}") ###DEBUG

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class KRSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = KRNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###KR

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###KR
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class KRNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):

        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_hidden_channels = 128 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_scale = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.last_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.last_hidden_channels, self.last_hidden_channels),
        ) ###KR

        torch.nn.init.xavier_uniform_(self.embedding_to_scale[-1].weight) ###KR
        torch.nn.init.zeros_(self.embedding_to_scale[-1].bias) ###KR

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in KR IS2024 used last (two) Resnet block, this is implementation of last one
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        num_output_blocks = len(self.output_blocks)
        for idx, (residual, block) in enumerate(zip(reversed(history), self.output_blocks)):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            # print("idx:", idx, "hidden shape:", hidden.shape)###DEBUG

            # Apply FiLM only at the LAST output ResBlock
            if idx == num_output_blocks - 1 and spkr_embedding is not None:
                # print("idx:", idx)###DEBUG
                spkr_embedding = spkr_embedding.to(hidden.dtype)
                scale = self.embedding_to_scale(spkr_embedding)
                # print("scale shape:", scale.shape) ###DEBUG
                # hidden = hidden + scale[:, :, None, None]
                hidden = hidden * scale[:, :, None, None]

            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class KRAddSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = KRAddNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###KR

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###KR
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class KRAddNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 256, ###DEBUG 1024 for XLSR, 256 for ECAPA
        **_,
    ):

        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_hidden_channels = 128 # according to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 256 ###DEBUG 1024 for XLSR, 256 for ECAPA

        self.embedding_to_scale = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.last_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.last_hidden_channels, self.last_hidden_channels),
        ) ###KR

        torch.nn.init.xavier_uniform_(self.embedding_to_scale[-1].weight) ###KR
        torch.nn.init.zeros_(self.embedding_to_scale[-1].bias) ###KR

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in KR IS2024 used last (two) Resnet block, this is implementation of last one
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        num_output_blocks = len(self.output_blocks)
        for idx, (residual, block) in enumerate(zip(reversed(history), self.output_blocks)):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            # print("idx:", idx, "hidden shape:", hidden.shape)###DEBUG

            # Apply FiLM only at the LAST output ResBlock
            if idx == num_output_blocks - 1 and spkr_embedding is not None:
                # print("idx:", idx)###DEBUG
                spkr_embedding = spkr_embedding.to(hidden.dtype)
                scale = self.embedding_to_scale(spkr_embedding)
                # print("scale shape:", scale.shape) ###DEBUG
                hidden = hidden + scale[:, :, None, None]
                # hidden = hidden * scale[:, :, None, None]

            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class KRFiLMSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = KRFiLMNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###KR

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###NTT
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class KRFiLMNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):
 
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_hidden_channels = 128 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_film = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.last_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.last_hidden_channels, 2 * self.last_hidden_channels),
        ) ###KR

        torch.nn.init.xavier_uniform_(self.embedding_to_film[-1].weight) ###KR
        torch.nn.init.zeros_(self.embedding_to_film[-1].bias) ###KR

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in NTT used after 1st Resnet block
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()


        # upsampling path
        to_project = []
        num_output_blocks = len(self.output_blocks)
        for idx, (residual, block) in enumerate(zip(reversed(history), self.output_blocks)):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            # print("idx:", idx, "hidden shape:", hidden.shape)###DEBUG

            # Apply FiLM only at the LAST output ResBlock
            if idx == num_output_blocks - 1 and spkr_embedding is not None:
                # print("idx:", idx)###DEBUG
                spkr_embedding = spkr_embedding.to(hidden.dtype)
                scale, shift = self.embedding_to_film(spkr_embedding).chunk(2, dim=1)
                # print("scale shape:", scale.shape, "shift shape:", shift.shape) ###DEBUG
                hidden = hidden * scale[:, :, None, None] + shift[:, :, None, None]

            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)


        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class MIDSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = MIDNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###MID

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###MID
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class MIDNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):

        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = 256 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_scale = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bottleneck_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
        ) ###MID

        torch.nn.init.xavier_uniform_(self.embedding_to_scale[-1].weight) ###MID
        torch.nn.init.zeros_(self.embedding_to_scale[-1].bias) ###MID

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in KR IS2024 used last (two) Resnet block, this is implementation of last one
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()
            else:
                # Apply FiLM at the bottleneck
                if spkr_embedding is not None:
                    spkr_embedding = spkr_embedding.to(hidden.dtype)
                    scale = self.embedding_to_scale(spkr_embedding)
                    # hidden = hidden + scale[:, :, None, None]
                    hidden = hidden * scale[:, :, None, None]

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)       

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class MIDAddSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = MIDAddNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###MID

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###MID
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class MIDAddNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):

        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = 256 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_scale = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bottleneck_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
        ) ###MID

        torch.nn.init.xavier_uniform_(self.embedding_to_scale[-1].weight) ###MID
        torch.nn.init.zeros_(self.embedding_to_scale[-1].bias) ###MID   

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in KR IS2024 used last (two) Resnet block, this is implementation of last one
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()
            else:
                # Apply FiLM at the bottleneck
                if spkr_embedding is not None:
                    spkr_embedding = spkr_embedding.to(hidden.dtype)
                    scale = self.embedding_to_scale(spkr_embedding)
                    hidden = hidden + scale[:, :, None, None]
                    # hidden = hidden * scale[:, :, None, None]

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)       

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length


class MIDFiLMSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = MIDFiLMNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###MID

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###NTT
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class MIDFiLMNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):
 
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = 256 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = 1024 ###DEBUG

        self.embedding_to_film = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.bottleneck_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.bottleneck_channels, 2 * self.bottleneck_channels),
        ) ###MID

        torch.nn.init.xavier_uniform_(self.embedding_to_film[-1].weight) ###MID
        torch.nn.init.zeros_(self.embedding_to_film[-1].bias) ###MID

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in NTT used after 1st Resnet block
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()
            else:
                # Apply FiLM at the bottleneck
                if spkr_embedding is not None:
                    spkr_embedding = spkr_embedding.to(hidden.dtype)
                    scale, shift = self.embedding_to_film(spkr_embedding).chunk(2, dim=1)
                    hidden = hidden * scale[:, :, None, None] + shift[:, :, None, None]
                    # print("Using speaker embedding in the bottleneck ")###DEBUG

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)

        to_project.append(hidden)


        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class ALLSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = ALLNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###ALL

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###MID
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class ALLNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):

        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.bottleneck_channels = 256 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = embedding_dim ###DEBUG

        # Create FiLM layers for each unique channel size
        unique_channels = sorted(set(self.channels)) ###ALL
        self.embedding_to_scale_layers = torch.nn.ModuleDict() ###ALL
        for ch in unique_channels:
            self.embedding_to_scale_layers[str(ch)] = torch.nn.Sequential(
                torch.nn.Linear(1024, ch),
                torch.nn.SiLU(),
                torch.nn.Linear(ch, ch),  # scale 
            ) ###ALL
            torch.nn.init.xavier_uniform_(self.embedding_to_scale_layers[str(ch)][-1].weight) ###ALL
            torch.nn.init.zeros_(self.embedding_to_scale_layers[str(ch)][-1].bias) ###ALL

        

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in KR IS2024 used last (two) Resnet block, this is implementation of last one
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                
                # Apply FiLM after each res_block
                if spkr_embedding is not None:
                    spkr_embedding_typed = spkr_embedding.to(hidden.dtype)
                    channel_size = str(hidden.shape[1])
                    if channel_size in self.embedding_to_scale_layers:
                        scale_layer = self.embedding_to_scale_layers[channel_size]
                        scale = scale_layer(spkr_embedding_typed)
                        hidden = hidden * scale[:, :, None, None]
                        # print(f"!down channel_size {channel_size}")

                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)
            

            # Apply FiLM after each res_block
            if spkr_embedding is not None:
                spkr_embedding_typed = spkr_embedding.to(hidden.dtype)
                channel_size = str(hidden.shape[1])
                if channel_size in self.embedding_to_scale_layers:
                    scale_layer = self.embedding_to_scale_layers[channel_size]
                    scale = scale_layer(spkr_embedding_typed)
                    hidden = hidden * scale[:, :, None, None]
                    # print(f"!up channel_size {channel_size}")

        to_project.append(hidden)    

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length

class ALLAddSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = ALLAddNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###ALL

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###MID
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class ALLAddNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):

        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.bottleneck_channels = 256 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = embedding_dim ###DEBUG

        # Create FiLM layers for each unique channel size
        unique_channels = sorted(set(self.channels)) ###ALL
        self.embedding_to_scale_layers = torch.nn.ModuleDict() ###ALL
        for ch in unique_channels:
            self.embedding_to_scale_layers[str(ch)] = torch.nn.Sequential(
                torch.nn.Linear(1024, ch),
                torch.nn.SiLU(),
                torch.nn.Linear(ch, ch),  # scale 
            ) ###ALL
            torch.nn.init.xavier_uniform_(self.embedding_to_scale_layers[str(ch)][-1].weight) ###ALL
            torch.nn.init.zeros_(self.embedding_to_scale_layers[str(ch)][-1].bias) ###ALL

        

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shape (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in KR IS2024 used last (two) Resnet block, this is implementation of last one
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                
                # Apply FiLM after each res_block
                if spkr_embedding is not None:
                    spkr_embedding_typed = spkr_embedding.to(hidden.dtype)
                    # print(f"spkr_embedding_typed {spkr_embedding_typed}") ###DEBUG-XLSR
                    channel_size = str(hidden.shape[1])
                    if channel_size in self.embedding_to_scale_layers:
                        # print(f"!!!down channel_size {channel_size}") ###DEBUG-XLSR
                        scale_layer = self.embedding_to_scale_layers[channel_size]
                        # print(f"scale layer {scale_layer}") ###DEBUG-XLSR
                        scale = scale_layer(spkr_embedding_typed)
                        # print(f"scale {scale}") ###DEBUG-XLSR
                        hidden = hidden + scale[:, :, None, None]
                        # print(f"hidden {hidden}") ###DEBUG-XLSR

                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)
            

            # Apply FiLM after each res_block
            if spkr_embedding is not None:
                spkr_embedding_typed = spkr_embedding.to(hidden.dtype)
                channel_size = str(hidden.shape[1])
                if channel_size in self.embedding_to_scale_layers:
                    scale_layer = self.embedding_to_scale_layers[channel_size]
                    scale = scale_layer(spkr_embedding_typed)
                    hidden = hidden + scale[:, :, None, None]
                    # print(f"!!!up channel_size {channel_size}")

        to_project.append(hidden)       

        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length


class ALLFiLMSpectrogramNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """This model handles complex-valued inputs by stacking real and imaginary components.
    Stacked tensor is processed using NCSN++ and the output is projected to generate real
    and imaginary components of the output channels.

    Args:
        in_channels: number of input complex-valued channels
        out_channels: number of output complex-valued channels
    """

    def __init__(self, *, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        # Number of input signals for this estimator
        if in_channels < 1:
            raise ValueError(
                f'Number of input channels needs to be larger or equal to one, current value {in_channels}'
            )

        self.in_channels = in_channels

        # Number of output signals for this estimator
        if out_channels < 1:
            raise ValueError(
                f'Number of output channels needs to be larger or equal to one, current value {out_channels}'
            )

        self.out_channels = out_channels

        # Instantiate noise conditional score network NCSN++
        ncsnpp_params = kwargs.copy()
        ncsnpp_params['in_channels'] = ncsnpp_params['out_channels'] = 2 * self.in_channels  # stack real and imag
        self.ncsnpp = ALLFiLMNoiseConditionalScoreNetworkPlusPlus(**ncsnpp_params) ###ALL

        # Output projection to generate real and imaginary components of the output channels
        self.output_projection = torch.nn.Conv2d(
            in_channels=2 * self.in_channels, out_channels=2 * self.out_channels, kernel_size=1
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:  %s', self.in_channels)
        logging.debug('\tout_channels: %s', self.out_channels)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(self, input, input_length=None, condition=None, spkr_embedding=None):
        # Stack real and imaginary components
        B, C_in, D, T = input.shape

        if C_in != self.in_channels:
            raise RuntimeError(f'Unexpected input channel size {C_in}, expected {self.in_channels}')

        # Stack real and imaginary parts
        input_real_imag = torch.stack([input.real, input.imag], dim=2)
        input = einops.rearrange(input_real_imag, 'B C RI F T -> B (C RI) F T')

        # Process using NCSN++ ###NTT
        output, output_length = self.ncsnpp(input=input, input_length=input_length, condition=condition, spkr_embedding = spkr_embedding)

        # Output projection
        output = self.output_projection(output)

        # Convert to complex-valued signal
        output = output.reshape(B, 2, self.out_channels, D, T)
        # Move real/imag dimension to the end
        output = output.permute(0, 2, 3, 4, 1)
        output = torch.view_as_complex(output.contiguous())

        return output, output_length

class ALLFiLMNoiseConditionalScoreNetworkPlusPlus(NeuralModule):
    """Implementation of Noise Conditional Score Network (NCSN++) architecture.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        nonlinearity: str = "swish",
        in_channels: int = 2,  # number of channels in the input image
        out_channels: int = 2,  # number of channels in the output image
        channels: Sequence[int] = (128, 128, 256, 256, 256),  # number of channels at start + at every resolution
        num_res_blocks: int = 2,
        num_resolutions: int = 4,
        init_scale: float = 1e-5,
        conditioned_on_time: bool = False,
        fourier_embedding_scale: float = 16.0,
        dropout_rate: float = 0.0,
        pad_time_to: Optional[int] = None,
        pad_dimension_to: Optional[int] = None,
        embedding_dim = 1024, ###DEBUG
        **_,
    ):
 
        super().__init__()

        # same nonlinearity is used throughout the whole network
        self.activation: torch.nn.Module = activation_registry[nonlinearity]()
        self.init_scale: float = init_scale

        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear")
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.bottleneck_channels = 256 # accourding to printout OR self.hidden_channels[0]-> did not test yet
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = num_resolutions
        self.conditioned_on_time = conditioned_on_time
        self.embedding_dim = embedding_dim ###DEBUG

        # self.embedding_to_film = torch.nn.Sequential(
        #     torch.nn.Linear(self.embedding_dim, self.bottleneck_channels),
        #     torch.nn.SiLU(),
        #     torch.nn.Linear(self.bottleneck_channels, 2 * self.bottleneck_channels),
        # ) ###MID

        # torch.nn.init.xavier_uniform_(self.embedding_to_film[-1].weight) ###MID
        # torch.nn.init.zeros_(self.embedding_to_film[-1].bias) ###MID

        # In __init__, replace the single embedding_to_film with a ModuleDict:
        # Create FiLM layers for each unique channel size
        unique_channels = sorted(set(self.channels)) ###ALL
        self.embedding_to_film_layers = torch.nn.ModuleDict() ###ALL
        for ch in unique_channels:
            self.embedding_to_film_layers[str(ch)] = torch.nn.Sequential(
                torch.nn.Linear(1024, ch),
                torch.nn.SiLU(),
                torch.nn.Linear(ch, 2 * ch),  # scale and shift
            ) ###ALL
            torch.nn.init.xavier_uniform_(self.embedding_to_film_layers[str(ch)][-1].weight) ###ALL
            torch.nn.init.zeros_(self.embedding_to_film_layers[str(ch)][-1].bias) ###ALL

        

        # padding setup
        self.pad_time_to = pad_time_to or 2**self.num_resolutions
        self.pad_dimension_to = pad_dimension_to or 2**self.num_resolutions

        if self.conditioned_on_time:
            self.time_embedding = torch.nn.Sequential(
                GaussianFourierProjection(embedding_size=self.channels[0], scale=fourier_embedding_scale),
                torch.nn.Linear(self.channels[0] * 2, self.channels[0] * 4),
                self.activation,
                torch.nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
            )

        self.input_pyramid = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.input_pyramid.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=ch, kernel_size=1))

        # each block takes an image and outputs an image
        # possibly changes number of channels
        # output blocks ("reverse" path of the unet) reuse outputs of input blocks ("forward" path)
        # so great care must be taken to in/out channels of each block
        # resolutions are handled in `forward`
        block_params = {
            "activation": self.activation,
            "dropout_rate": dropout_rate,
            "init_scale": self.init_scale,
            "diffusion_step_embedding_dim": channels[0] * 4 if self.conditioned_on_time else None,
        }
        self.input_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(self.channels[:-1], self.channels[1:]):
            for n in range(num_res_blocks):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch if n == 0 else out_ch, out_ch=out_ch, **block_params)
                self.input_blocks.append(block)

        self.output_blocks = torch.nn.ModuleList()
        for in_ch, out_ch in zip(reversed(self.channels[1:]), reversed(self.channels[:-1])):
            for n in reversed(range(num_res_blocks)):
                block = ResnetBlockBigGANPlusPlus(in_ch=in_ch, out_ch=out_ch if n == 0 else in_ch, **block_params)
                self.output_blocks.append(block)

        self.projection_blocks = torch.nn.ModuleList()
        for ch in self.channels[:-1]:
            self.projection_blocks.append(torch.nn.Conv2d(ch, out_channels, kernel_size=1))

        assert len(self.input_pyramid) == self.num_resolutions
        assert len(self.input_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.output_blocks) == self.num_resolutions * self.num_res_blocks
        assert len(self.projection_blocks) == self.num_resolutions

        self.init_weights_()

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tin_channels:         %s', self.in_channels)
        logging.debug('\tout_channels:        %s', self.out_channels)
        logging.debug('\tchannels:            %s', self.channels)
        logging.debug('\tnum_res_blocks:      %s', self.num_res_blocks)
        logging.debug('\tnum_resolutions:     %s', self.num_resolutions)
        logging.debug('\tconditioned_on_time: %s', self.conditioned_on_time)
        logging.debug('\tpad_time_to:         %s', self.pad_time_to)
        logging.debug('\tpad_dimension_to:    %s', self.pad_dimension_to)

    def init_weights_(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # torch.nn submodules with scaled init
        for module in self.projection_blocks:
            torch.nn.init.xavier_uniform_(module.weight, gain=self.init_scale)

        # non-torch.nn submodules can have their own init schemes
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "init_weights_"):
                module.init_weights_()

    @typecheck(
        input_types={
            "input": NeuralType(('B', 'C', 'D', 'T')),
        },
        output_types={
            "output": NeuralType(('B', 'C', 'D', 'T')),
        },
    )
    def pad_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pad input tensor to match the required dimensions across `T` and `D`."""
        *_, D, T = input.shape
        output = input

        # padding across time
        if T % self.pad_time_to != 0:
            output = F.pad(output, (0, self.pad_time_to - T % self.pad_time_to))

        # padding across dimension
        if D % self.pad_dimension_to != 0:
            output = F.pad(output, (0, 0, 0, self.pad_dimension_to - D % self.pad_dimension_to))

        return output

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "input_length": NeuralType(('B',), LengthsType(), optional=True),
            "condition": NeuralType(('B',), FloatType(), optional=True),
            "spkr_embedding": NeuralType(('B', 'C'), VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "output_length": NeuralType(('B',), LengthsType(), optional=True),
        }

    @typecheck()
    def forward(
        self, *, input: torch.Tensor, input_length: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None, spkr_embedding: Optional[torch.Tensor] = None
    ):
        """Forward pass of the model.

        Args:
            input: input tensor, shjae (B, C, D, T)
            input_length: length of the valid time steps for each example in the batch, shape (B,)
            condition: scalar condition (time) for the model, will be embedded using `self.time_embedding`
            spkr_embedding: speaker embedding e.g. in NTT used after 1st Resnet block
        """
        assert input.shape[1] == self.in_channels

        # apply padding at the input
        *_, D, T = input.shape
        input = self.pad_input(input=input)

        if input_length is None:
            # assume all time frames are valid
            input_length = torch.LongTensor([input.shape[-1]] * input.shape[0]).to(input.device)

        lengths = input_length

        if condition is not None:
            if len(condition.shape) != 1:
                raise ValueError(
                    f"Expected conditon to be a 1-dim tensor, got a {len(condition.shape)}-dim tensor of shape {tuple(condition.shape)}"
                )
            if condition.shape[0] != input.shape[0]:
                raise ValueError(
                    f"Condition {tuple(condition.shape)} and input {tuple(input.shape)} should match along the batch dimension"
                )

            condition = self.time_embedding(torch.log(condition))
        # if spkr_embedding is not None:
        #     print("Using speaker embedding in NTTNoiseConditionalScoreNetworkPlusPlus...")
        # downsample and project input image to add later in the downsampling path
        pyramid = [input]
        for resolution_num in range(self.num_resolutions - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        pyramid = [block(image) for image, block in zip(pyramid, self.input_pyramid)]

        # downsampling path
        history = []
        hidden = torch.zeros_like(pyramid[0])
        input_blocks = iter(self.input_blocks)
        for resolution_num, image in enumerate(pyramid):
            hidden = (hidden + image) / math.sqrt(2.0)
            hidden = mask_sequence_tensor(hidden, lengths)

            for _ in range(self.num_res_blocks):
                hidden = next(input_blocks)(hidden, condition)
                hidden = mask_sequence_tensor(hidden, lengths)
                
                # Apply FiLM after each res_block
                if spkr_embedding is not None:
                    spkr_embedding_typed = spkr_embedding.to(hidden.dtype)
                    channel_size = str(hidden.shape[1])
                    if channel_size in self.embedding_to_film_layers:
                        film_layer = self.embedding_to_film_layers[channel_size]
                        scale, shift = film_layer(spkr_embedding_typed).chunk(2, dim=1)
                        hidden = hidden * (1+scale[:, :, None, None]) + shift[:, :, None, None]
                
                history.append(hidden)

            final_resolution = resolution_num == self.num_resolutions - 1
            if not final_resolution:
                hidden = self.downsample(hidden)
                lengths = (lengths / 2).ceil().long()

        # upsampling path
        to_project = []
        for residual, block in zip(reversed(history), self.output_blocks):
            if hidden.shape != residual.shape:
                to_project.append(hidden)
                hidden = self.upsample(hidden)
                lengths = (lengths * 2).long()

            hidden = (hidden + residual) / math.sqrt(2.0)
            hidden = block(hidden, condition)
            hidden = mask_sequence_tensor(hidden, lengths)
            
            # Apply FiLM after each res_block in upsampling path
            if spkr_embedding is not None:
                spkr_embedding_typed = spkr_embedding.to(hidden.dtype)
                channel_size = str(hidden.shape[1])
                if channel_size in self.embedding_to_film_layers:
                    film_layer = self.embedding_to_film_layers[channel_size]
                    scale, shift = film_layer(spkr_embedding_typed).chunk(2, dim=1)
                    hidden = hidden * scale[:, :, None, None] + shift[:, :, None, None]

        to_project.append(hidden)        
        # projecting to images
        images = []
        for tensor, projection in zip(to_project, reversed(self.projection_blocks)):
            image = projection(tensor)
            images.append(F.interpolate(image, size=input.shape[-2:]))  # TODO write this loop using self.upsample

        result = sum(images)

        assert result.shape[-2:] == input.shape[-2:]

        # remove padding
        result = result[:, :, :D, :T]
        return result, input_length


class GaussianFourierProjection(NeuralModule):
    """Gaussian Fourier embeddings for input scalars.

    The input scalars are typically time or noise levels.
    """

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "input": NeuralType(('B',), FloatType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'D'), VoidType()),
        }

    def forward(self, input):
        x_proj = input[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResnetBlockBigGANPlusPlus(torch.nn.Module):
    """Implementation of a ResNet block for the BigGAN model.

    References:
        - Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, NeurIPS 2021
        - Brock et al., Large scale GAN training for high fidelity natural image synthesis, ICLR 2018
    """

    def __init__(
        self,
        activation: torch.nn.Module,
        in_ch: int,
        out_ch: int,
        diffusion_step_embedding_dim: Optional[int] = None,
        init_scale: float = 1e-5,
        dropout_rate: float = 0.1,
        in_num_groups: Optional[int] = None,
        out_num_groups: Optional[int] = None,
        eps: float = 1e-6,
    ):
        """
        Args:
            activation (torch.nn.Module): activation layer (ReLU, SiLU, etc)
            in_ch (int): number of channels in the input image
            out_ch (int, optional): number of channels in the output image
            diffusion_step_embedding_dim (int, optional): dimension of diffusion timestep embedding. Defaults to None (no embedding).
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
            init_scale (float, optional): scaling for weight initialization. Defaults to 0.0.
            in_num_groups (int, optional): num_groups in the first GroupNorm. Defaults to min(in_ch // 4, 32)
            out_num_groups (int, optional): num_groups in the second GroupNorm. Defaults to min(out_ch // 4, 32)
            eps (float, optional): eps parameter of GroupNorms. Defaults to 1e-6.
        """
        super().__init__()
        in_num_groups = in_num_groups or min(in_ch // 4, 32)
        out_num_groups = out_num_groups or min(out_ch // 4, 32)

        self.init_scale = init_scale

        self.input_block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=in_num_groups, num_channels=in_ch, eps=eps),
            activation,
        )

        self.middle_conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        if diffusion_step_embedding_dim is not None:
            self.diffusion_step_projection = torch.nn.Sequential(
                activation,
                torch.nn.Linear(diffusion_step_embedding_dim, out_ch),
                einops.layers.torch.Rearrange("batch dim -> batch dim 1 1"),
            )

        self.output_block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=out_num_groups, num_channels=out_ch, eps=eps),
            activation,
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
        )

        if in_ch != out_ch:
            self.residual_projection = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

        self.act = activation
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.init_weights_()

    def init_weights_(self):
        """Weight initialization"""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # a single Conv2d is initialized with gain
        torch.nn.init.xavier_uniform_(self.output_block[-1].weight, gain=self.init_scale)

    def forward(self, x: torch.Tensor, diffusion_time_embedding: Optional[torch.Tensor] = None):
        """Forward pass of the model.

        Args:
            x: input tensor
            diffusion_time_embedding: embedding of the diffusion time step

        Returns:
            Output tensor
        """
        h = self.input_block(x)
        h = self.middle_conv(h)

        if diffusion_time_embedding is not None:
            h = h + self.diffusion_step_projection(diffusion_time_embedding)

        h = self.output_block(h)

        if x.shape != h.shape:  # matching number of channels
            x = self.residual_projection(x)
        return (x + h) / math.sqrt(2.0)

# hidden = hidden * torch.tanh(1.0 + scale) ### Model saved as 20260110
# hidden = hidden *  scale
# hidden = hidden + scale
### FILM style update
# self.embedding_to_film = nn.Linear(256, 2 * 128)
# scale, shift = self.embedding_to_film(spkr_embedding).chunk(2, dim=1)
# hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
