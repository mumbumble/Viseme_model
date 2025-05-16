import math
import typing

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def make_phoneme_feature(
    phoneme_feature: torch.Tensor,
    use_label: bool = False,
    return_extra_dict: bool = False,
) -> typing.Tuple[torch.Tensor, typing.Optional[typing.Dict[str, torch.Tensor]]]:
    return_dict = None
    phoneme_feature_list = [
        phoneme_feature,
        phoneme_softmax := F.softmax(phoneme_feature, dim=-1),
    ]

    if use_label:
        phoneme_feature_list.append(
            phoneme_label := F.one_hot(
                torch.argmax(phoneme_softmax, dim=2),
                num_classes=phoneme_feature.shape[-1],
            )
        )

    if return_extra_dict:
        return_dict = {
            "phn_probs": phoneme_feature,
            "phn_softmax": phoneme_softmax,
        }

        if use_label:
            return_dict["phn_labels"] = phoneme_label

    return torch.cat(phoneme_feature_list, dim=-1), return_dict


def sampling_length(wav_length: float, win_length: int, hop_length: float) -> int:
    """
    sampling_window의 간격은 hop_length이고,
    sampling_window의 크기는 win_length일 때
    sliding_window의 총 갯수에 대한 계산을 함
    """
    return math.ceil((wav_length - win_length + 1) / hop_length)


def sampling_length_even(wav_length: float, win_length: int, hop_length: float) -> int:
    """
    sampling_window의 간격은 hop_length이고,
    sampling_window의 크기는 win_length일 때
    sliding_window의 총 갯수에 대한 계산을 함
    """

    sp_length = (wav_length - win_length) // hop_length + 1
    if sp_length % 2 == 0:
        return sp_length
    else:  # vae encoder decoder 에서 짝수로 맞춰주느라고.......
        return sp_length + 1


def linear_interpolation(input_seq: torch.Tensor, output_len: int) -> torch.Tensor:
    """
    input_seq: Batch, input_len, feature_dim
    return: Batch, output_len, feature_dim
    """
    input_seq = input_seq.transpose(1, 2)
    output_seq = F.interpolate(input_seq, output_len, align_corners=True, mode="linear")
    return output_seq.transpose(1, 2)


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


# ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    wav: torch.Tensor = None,
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(
        real_samples.device
    )

    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)

    if wav is None:
        d_interpolates = discriminator(interpolates)
    else:
        d_interpolates = discriminator(wav, interpolates)

    fake = autograd.Variable(torch.ones_like(d_interpolates), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients, *_ = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
