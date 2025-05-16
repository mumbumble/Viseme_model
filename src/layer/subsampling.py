#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
import typing

import torch
import torch.nn as nn

from .embedding import PositionalEncoding


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length or 1/2 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param nn.Module pos_enc: custom position encoding layer

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc: nn.Module = None,
        subsample_by_2: bool = False,
    ):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.subsample_by_2 = subsample_by_2
        if subsample_by_2:
            self.conv = nn.Sequential(
                nn.Conv2d(1, odim, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(odim, odim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )
            self.out = nn.Sequential(
                nn.Linear(odim * (idim // 2), odim),
                pos_enc or PositionalEncoding(odim, dropout_rate),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, odim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(odim, odim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )
            self.out = nn.Sequential(
                nn.Linear(odim * (idim // 4), odim),
                pos_enc or PositionalEncoding(odim, dropout_rate),
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is not None:
            x_mask = x_mask[:, :, ::2]

            if not self.subsample_by_2:
                x_mask = x_mask[:, :, ::2]

        return x, x_mask

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dNoSubsampling(nn.Module):
    """Convolutional 2D without subsampling.

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param nn.Module pos_enc: custom position encoding layer

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc: nn.Module = None,
    ):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * idim, odim),
            pos_enc or PositionalEncoding(odim, dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        return x, x_mask

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 5, 3),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            PositionalEncoding(odim, dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is not None:
            x_mask = x_mask[:, :, :-2:2][:, :, :-4:3]

        return x, x_mask


class Conv2dSubsampling8(nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.conv(x.unsqueeze(1))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is not None:
            x_mask = x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

        return x, x_mask
