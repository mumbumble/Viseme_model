import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    inference_chunk_length: int = 512

    def __init__(
        self,
        input_features: int,
        recurrent_features: int,
        batch_first: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_features,
            recurrent_features,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature, _ = self.rnn(x)

        return feature

        # if self.training:
        #     return self.rnn(x)[0]
        # else:
        #     # evaluation mode: support for longer sequences that do not fit in memory
        #     batch_size, sequence_length, input_features = x.shape
        #     hidden_size = self.rnn.hidden_size
        #     num_directions = 2 if self.rnn.bidirectional else 1
        #
        #     h = torch.zeros(num_directions, batch_size, hidden_size).to(x.device)
        #     c = torch.zeros(num_directions, batch_size, hidden_size).to(x.device)
        #     output = torch.zeros(
        #         batch_size, sequence_length, num_directions * hidden_size
        #     ).to(x.device)
        #
        #     # forward direction
        #     slices = range(0, sequence_length, self.inference_chunk_length)
        #     for start in slices:
        #         end = start + self.inference_chunk_length
        #         output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))
        #
        #     # reverse direction
        #     if self.rnn.bidirectional:
        #         h.zero_()
        #         c.zero_()
        #
        #         for start in reversed(slices):
        #             end = start + self.inference_chunk_length
        #             result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
        #             output[:, start:end, hidden_size:] = result[:, :, hidden_size:]
        #
        #     return output
