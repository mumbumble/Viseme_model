import typing

import torch
import torch.nn as nn

from nface.atl.utils import Beholder

from layer import (
    VGG2L,
    Conv1dLinear,
    Conv2dNoSubsampling,
    Conv2dSubsampling,
    ConvolutionModule,
    EncoderLayer,
    MultiHeadedAttention,
    MultiLayeredConv1d,
    PositionalEncoding,
    PositionwiseFeedForward,
    RelPositionalEncoding,
    RelPositionMultiHeadedAttention,
    ScaledPositionalEncoding,
    repeat,
)
from layer.utils import make_pad_mask


class Encoder(nn.Module, metaclass=Beholder):
    pass


class ConformerEncoder(Encoder):
    """Conformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_pos_enc_layer_type: encoder positional encoding layer type
    :param str encoder_attn_layer_type: encoder attention layer type
    :param str activation_type: encoder activation function type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int = 80,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: typing.Optional[
            typing.Union[
                nn.Module,
                typing.Literal["linear", "conv2d", "vgg2l", "embed"],
            ]
        ] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: typing.Literal[
            "linear", "conv1d", "conv1d-linear"
        ] = "linear",
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        pos_enc_layer_type: typing.Literal[
            "abs_pos", "scaled_abs_pos", "rel_pos"
        ] = "abs_pos",
        selfattention_layer_type: typing.Literal[
            "selfattn", "rel_selfattn"
        ] = "selfattn",
        activation_type: typing.Literal["swish", "relu", "selu", "hardtanh"] = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        no_subsample: bool = False,
        subsample_by_2: bool = False,
    ):
        """Construct an Encoder object."""
        super().__init__()

        self._output_size = attention_dim

        if activation_type == "swish":
            activation = torch.nn.SiLU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "selu":
            activation = torch.nn.SELU()
        elif activation_type == "hardtanh":
            activation = torch.nn.Hardtanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError(f"unknown pos_enc_layer_type: {pos_enc_layer_type}")

        if input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(input_size, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            if no_subsample:
                self.embed = Conv2dNoSubsampling(
                    input_size,
                    attention_dim,
                    dropout_rate,
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
            else:
                self.embed = Conv2dSubsampling(
                    input_size,
                    attention_dim,
                    dropout_rate,
                    pos_enc_class(attention_dim, positional_dropout_rate),
                    subsample_by_2,  # NOTE(Sx): added by songxiang
                )
        elif input_layer == "vgg2l":
            self.embed = VGG2L(input_size, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"unknown input_layer: {input_layer}")

        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError(
                f"Support only linear or conv1d, but got {positionwise_layer_type}"
            )

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        else:
            raise ValueError(f"Unknown encoder_attn_layer: {selfattention_layer_type}")

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim, eps=1e-12)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """
        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input lengths (B)
            prev_states: Not to be used now.
        Returns:
            Position embedded tensor and mask
        """
        masks = ~make_pad_mask(ilens).unsqueeze(1).to(xs_pad.device)

        if isinstance(self.embed, (Conv2dSubsampling, Conv2dNoSubsampling, VGG2L)):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if isinstance(xs_pad, tuple):
            xs_pad, *_ = xs_pad

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.sum(axis=-1)

        return xs_pad, olens, None
