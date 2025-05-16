import math
import typing

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    :param reverse: whether to reverse the input position

    """

    def __init__(
        self,
        d_model,
        dropout_rate,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.reverse = reverse
        self.dropout = nn.Dropout(p=dropout_rate)

        position = (
            torch.arange(max_len - 1, -1, -1.0) if reverse else torch.arange(max_len)
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

        def load_state_dict_pre_hook(
            state_dict: dict,
            prefix: str,
            local_metadata: dict,
            strict: bool,
            missing_keys: typing.List[str],
            unexpected_keys: typing.List[str],
            error_msgs: typing.List[str],
        ):
            """Load state dict pre hook."""
            if (key := f"{prefix}pe") in state_dict:
                del state_dict[key]

        self._register_load_state_dict_pre_hook(load_state_dict_pre_hook)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)
        return self.pe[:, : x.size(1)]

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        x += self.embedding(x)
        x = self.dropout(x)

        return x


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf
    """

    def __init__(
        self,
        d_model,
        dropout_rate,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super().__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            reverse=reverse,
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        x += self.alpha * self.embedding(x)
        x = self.dropout(x)

        return x


class RelPositionalEncoding(PositionalEncoding):
    """Relitive positional encoding module.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(
        self,
        d_model,
        dropout_rate,
        max_len: int = 5000,
        reverse: bool = True,
    ):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length
        :param bool reverse: whether to reverse the input position

        """
        super().__init__(d_model, dropout_rate, max_len=max_len, reverse=reverse)
        self.xscale = math.sqrt(self.d_model)

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: x. Its shape is (batch, time, ...)
            torch.Tensor: pos_emb. Its shape is (1, time, ...)

        """
        x *= self.xscale
        pos_emb = self.embedding(x)

        return self.dropout(x), self.dropout(pos_emb)
