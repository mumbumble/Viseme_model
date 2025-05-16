#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""

import typing

import torch.nn as nn


class MultiSequential(nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)

        return args


def repeat(times: int, fn: typing.Callable[[int], nn.Module]) -> MultiSequential:
    """Repeat module N times.

    :param int times: repeat time
    :param function fn: function to generate module
    :return: repeated module
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn(n) for n in range(times)])
