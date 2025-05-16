import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: nn.Module = None,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # GLU mechanism
        x = self.pointwise_conv1(x.transpose(1, 2))  # (batch, 2*channel, dim)
        x = F.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)


class ConvStack(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)

        return x


class CustomConv1d(nn.Module):
    """
    kernel_size 가 항상 홀수 이고, 항상 out_channel 과 in_channel이 동일한 구조
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int):
        super(CustomConv1d, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"Invalid Kernel Size {kernel_size}")

        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class Conv1dNormReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: typing.Union[str, int] = 1,
        bias: bool = True,
        norm: typing.Union[
            typing.Literal["batch", "instance", "lazy_batch", "lazy_instance"],
            nn.Module,
            None,
        ] = "batch",
        activation: typing.Union[
            typing.Literal["relu", "leaky_relu"],
            nn.Module,
            None,
        ] = "relu",
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        if isinstance(norm, str):
            norm = getattr(nn, f"{''.join(map(str.capitalize, norm.split('_')))}Norm1d")
        if norm is not None:
            norm = norm(out_channels)
        self.norm = norm

        if isinstance(activation, str):
            activation = {
                "relu": nn.ReLU,
                "leaky_relu": nn.LeakyReLU,
            }[activation]
        if activation is not None:
            activation = activation()
        self.activation = activation

    @classmethod
    def to_sequence(cls, *args, **kwargs) -> nn.Module:
        instance = cls(*args, **kwargs)

        return nn.Sequential(
            instance.conv1d,
            instance.norm,
            instance.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x
