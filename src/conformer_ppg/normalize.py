import typing

import torch
import torch.nn as nn

from nface.atl.utils import Beholder

def make_pad_mask(
    lengths: typing.Union[torch.Tensor, typing.List[int]],
    xs: typing.Optional[torch.Tensor] = None,
    length_dim: int = -1,
) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    bs = len(lengths)
    size = lengths.max().item() if xs is None else xs.size(length_dim)

    seq_range_expand: torch.Tensor = torch.arange(0, size).unsqueeze(0).expand(bs, size)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        if length_dim < 0:
            length_dim += xs.dim()

        if mask.ndim <= length_dim + 1:
            for _ in range(mask.ndim, xs.dim()):
                mask = mask.unsqueeze(-1)

        else:
            raise ValueError(
                "length_dim: {} cannot be larger then mask.ndim: {}".format(
                    length_dim, mask.ndim
                )
            )

        mask = mask.expand_as(xs).to(xs.device)

    return mask

class Normalize(nn.Module, metaclass=Beholder):
    pass


class UtteranceMVN(Normalize):
    def __init__(
        self,
        norm_means: bool = True,
        norm_vars: bool = False,
        eps: float = 1.0e-20,
    ):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def extra_repr(self):
        return f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)

        """
        return utterance_mvn(
            x,
            ilens,
            norm_means=self.norm_means,
            norm_vars=self.norm_vars,
            eps=self.eps,
        )


def utterance_mvn(
    x: torch.Tensor,
    ilens: torch.Tensor = None,
    norm_means: bool = True,
    norm_vars: bool = False,
    eps: float = 1.0e-20,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Apply utterance mean and variance normalization

    Args:
        x: (B, T, D), assumed zero padded
        ilens: (B,)
        norm_means:
        norm_vars:
        eps:

    """
    if ilens is None:
        ilens = x.new_full([x.size(0)], x.size(1))

    ilens_ = ilens.to(x.device)
    for _ in range(x.dim() - 1):
        ilens_ = ilens_.unsqueeze(-1)

    if x.requires_grad:
        x = x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)

    else:
        x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)

    mean = x.sum(dim=1, keepdim=True) / ilens_

    if norm_means:
        x -= mean

        if norm_vars:
            var = x.pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            x = x / std.sqrt()
        return x, ilens

    if norm_vars:
        y = x - mean
        y.masked_fill_(make_pad_mask(ilens, y, 1), 0.0)
        var = y.pow(2).sum(dim=1, keepdim=True) / ilens_
        std = torch.clamp(var.sqrt(), min=eps)
        x /= std
    return x, ilens
