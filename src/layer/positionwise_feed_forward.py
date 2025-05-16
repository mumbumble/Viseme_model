#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Module = None,
    ):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward funciton."""
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x
