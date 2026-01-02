#  Copyright 2025 ETH Zurich
#  Created by Fan Yang, Robotic Systems Lab, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""3D Positional Encoding for volumetric features.

This code is adapted from:
https://github.com/tatp22/multidim-positional-encoding
"""

import math

import torch
import torch.nn as nn


def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """Gets a base embedding for one dimension with sin and cos intertwined."""
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for volumetric tensors.

    Args:
        channels: The last dimension of the tensor you want to apply pos emb to.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.channels = channels

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: A 5d tensor of size (batch_size, x, y, z, ch)

        Returns:
            Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].expand(batch_size, -1, -1, -1, -1)


class PositionalEncodingPermute3D(nn.Module):
    """3D positional encoding for channel-first tensors.

    Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self) -> int:
        return self.penc.org_channels
