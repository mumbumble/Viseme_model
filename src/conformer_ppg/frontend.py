import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nface.atl.utils import Beholder
from nface.atl.utils.compat.librosa import librosa


from layer.utils import make_pad_mask


class Frontend(nn.Module, metaclass=Beholder):
    pass


class ASRFrontend(Frontend):
    """Conventional frontend structure for ASR

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: typing.Union[int, str] = 16000,
        n_fft: int = 1024,
        win_length: int = 800,
        hop_length: int = 160,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        norm: int = 1,
        kaldi_padding_mode: bool = False,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.downsample_rate = downsample_rate

        # Deepcopy (In general, dict shouldn't be used as default arg)
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            kaldi_padding_mode=kaldi_padding_mode,
        )

        self.n_mels = n_mels
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            norm=norm,
        )

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(inputs, input_lengths)
        
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = torch.complex(input_stft[..., 0], input_stft[..., 1])

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: torch.complex(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        # NOTE(sx): pad
        max_len = input_feats.size(1)
        if self.downsample_rate > 1 and max_len % self.downsample_rate != 0:
            padding = self.downsample_rate - max_len % self.downsample_rate
            input_feats = F.pad(input_feats, (0, 0, 0, padding), "constant", 0)
            feats_lens[torch.argmax(feats_lens)] = max_len + padding

        return input_feats, feats_lens


class Stft(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: typing.Union[int, None] = 512,
        hop_length: int = 128,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        kaldi_padding_mode=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.kaldi_padding_mode = kaldi_padding_mode
        if self.kaldi_padding_mode:
            self.win_length = 400

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"pad_mode={self.pad_mode}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self,
        inputs: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            inputs: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = inputs.size(0)
        if inputs.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            inputs = inputs.transpose(1, 2).reshape(-1, inputs.size(1))
        else:
            multi_channel = False

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if not self.kaldi_padding_mode:
            output = torch.view_as_real(
                torch.stft(
                    input=inputs,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    center=self.center,
                    pad_mode=self.pad_mode,
                    normalized=self.normalized,
                    onesided=self.onesided,
                    return_complex=True,
                )
            )

        else:
            # NOTE(sx): Use Kaldi-fasion padding, maybe wrong
            num_pads = self.n_fft - self.win_length
            inputs = F.pad(inputs, (num_pads, 0))
            output = torch.stft(
                inputs,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=False,
                pad_mode=self.pad_mode,
                normalized=self.normalized,
                onesided=self.onesided,
                return_complex=True,
            )
        print("## DEBUG SIFT:", output.shape)
        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens += 2 * pad

            olens = (
                torch.div(
                    ilens - self.win_length,
                    self.hop_length,
                    rounding_mode="trunc",
                )
                + 1
            )

            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)

            return output, olens

        return output, None


class LogMel(nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
        norm: {None, 1, np.inf} [scalar]
            if 1, divide the triangular mel weights by the width of the mel band
            (area normalization).  Otherwise, leave all the triangles aiming for
            a peak value of 1.0

    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        norm=1,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        self.mel_options = _mel_options

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        if librosa is not None:
            melmat = librosa.filters.mel(**_mel_options)
            # melmat: (D2, D1) -> (D1, D2)
            self.register_buffer("melmat", torch.from_numpy(melmat.T).float())
            inv_mel = np.linalg.pinv(melmat)
            self.register_buffer("inv_melmat", torch.from_numpy(inv_mel.T).float())
        # Default shape for melmat, inv_melmat for librosa is not installed
        else:
            self.register_buffer("melmat", torch.zeros((513, 80), dtype=torch.float))
            self.register_buffer(
                "inv_melmat", torch.zeros((80, 513), dtype=torch.float)
            )

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)

        logmel_feat = (mel_feat + 1e-20).log()
        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = torch.full(
                (feat.size(0),),
                feat.size(1),
                dtype=torch.int32,
            )

        return logmel_feat, ilens
