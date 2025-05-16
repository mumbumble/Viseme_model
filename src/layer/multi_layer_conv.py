#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer module for FFT block in FastSpeech (Feed-forward Transformer)."""

import torch
import torch.nn as nn


class MultiLayeredConv1d(nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transforner block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(
        self,
        in_chans: int,
        hidden_chans: int,
        kernel_size: int,
        dropout_rate: float,
    ):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = self.w_1(x.transpose(-1, 1))
        x = self.activation(x).transpose(-1, 1)
        x = self.dropout(x).transpose(-1, 1)
        x = self.w_2(x).transpose(-1, 1)
        return x


class Conv1dLinear(nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(
        self,
        in_chans: int,
        hidden_chans: int,
        kernel_size: int,
        dropout_rate: float,
    ):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = nn.Linear(hidden_chans, in_chans)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = self.w_1(x.transpose(-1, 1))
        x = torch.relu(x).transpose(-1, 1)
        x = self.w_2(self.dropout(x))
        return x
