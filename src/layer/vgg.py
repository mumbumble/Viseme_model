"""VGG2L definition for transformer-transducer."""
import typing

import torch
import torch.nn as nn


class VGG2L(nn.Module):
    """VGG2L module for transformer-transducer encoder."""

    def __init__(self, idim: int, odim: int):
        """Construct a VGG2L object.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs

        """
        super(VGG2L, self).__init__()

        self.vgg2l = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 2)),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.output = nn.Linear(128 * ((idim // 2) // 2), odim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """VGG2L forward for x.

        Args:
            x (torch.Tensor): input torch (B, T, idim)
            x_mask (torch.Tensor): (B, 1, T)

        Returns:
            x (torch.Tensor): input torch (B, sub(T), attention_dim)
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x = x.unsqueeze(1)
        x = self.vgg2l(x)

        b, c, t, f = x.size()

        x = self.output(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is not None:
            x_mask = self.create_new_mask(x_mask, x)

        return x, x_mask

    def create_new_mask(self, x_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Create a subsampled version of x_mask.

        Args:
            x_mask (torch.Tensor): (B, 1, T)
            x (torch.Tensor): (B, sub(T), attention_dim)

        Returns:
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x_t1 = x_mask.size(2) - (x_mask.size(2) % 3)
        x_mask = x_mask[:, :, :x_t1][:, :, ::3]

        x_t2 = x_mask.size(2) - (x_mask.size(2) % 2)
        x_mask = x_mask[:, :, :x_t2][:, :, ::2]

        return x_mask
