import typing
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from ruamel.yaml import YAML


class PPGModel(nn.Module):
    def __init__(
        self,
        frontend: nn.Module,
        normalize: nn.Module,
        encoder: nn.Module,
    ):
        super().__init__()
        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder

    @classmethod
    def from_static(
        cls,
        model_file: typing.Union[str, Path] = None,
        config_file: typing.Union[str, Path] = None,
        **kwargs,
    ) -> "PPGModel":
        static_path = Path(__file__).parent / "static"

        model_file = (
            static_path.joinpath("ppg-model.pth")
            if model_file is None
            else Path(model_file)
        )
        config_file = (
            static_path.joinpath("ppg-model.yml")
            if config_file is None
            else Path(config_file)
        )

        static_state_dict = torch.load(model_file, map_location="cpu")
        static_state_dict = OrderedDict(
            (k, v) for k, v in static_state_dict.items() if k.startswith("encoder.")
        )

        model = cls.from_config(config_file, **kwargs)
        model.load_state_dict(static_state_dict, strict=False)
        model.eval()
        model.requires_grad_(False)

        return model

    @classmethod
    def from_config(
        cls,
        config: typing.Union[str, Path, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> "PPGModel":
        if isinstance(config, (str, Path)):
            config = Path(config)
            yaml = YAML(typ="safe")
            with config.open() as fp:
                config = yaml.load(fp)

            return cls.from_config(config, **kwargs)

        config |= kwargs

        frontend_type = config.get("frontend")
        frontend_conf = config.get("frontend_conf")

        normalize_type = config.get("normalize")
        normalize_conf = config.get("normalize_conf")

        encoder_type = config.get("encoder")
        encoder_conf = config.get("encoder_conf")

        from .encoder import Encoder
        from .frontend import Frontend
        from .normalize import Normalize

        return cls(
            frontend=Frontend.get(frontend_type)(**frontend_conf),
            normalize=Normalize.get(normalize_type)(**normalize_conf),
            encoder=Encoder.get(encoder_type)(**encoder_conf),
        )

    def _extract_feature(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            return self.frontend(speech, speech_lengths)

        # No frontend and no feature extract
        return speech, speech_lengths

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        speech_lengths = torch.full(
            (speech.size(0),),
            speech.size(1),
            dtype=torch.int32,
        )
        # print("## DEBUG PPGmodelFORWARD1:", speech.shape, speech_lengths)
        feats, feats_lengths = self._extract_feature(speech, speech_lengths)
        # print("## DEBUG PPGmodelFORWARD2:", feats.shape, feats_lengths)
        feats, feats_lengths = self.normalize(feats, feats_lengths)
        # print("## DEBUG PPGmodelFORWARD3:", feats.shape, feats_lengths)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        # print("## DEBUG PPGmodelFORWARD4:", encoder_out.shape, encoder_out_lens)

        return encoder_out


class PPGPhonemeModel(nn.Module):
    def __init__(self, base_model: nn.Module, model_complexity: int, num_phoneme: int):
        super().__init__()

        self.base_model = base_model
        self.base_model.eval()

        self.phoneme_regressor = nn.Sequential(
            nn.Linear(model_complexity, num_phoneme),
        )

    @classmethod
    def from_static(
        cls,
        model_file: typing.Union[str, Path] = None,
        config_file: typing.Union[str, Path] = None,
        **kwargs,
    ) -> "PPGPhonemeModel":
        static_path = Path(__file__).parent / "static"

        model_file = (
            static_path.joinpath("ppg-phn-model.pth")
            if model_file is None
            else Path(model_file)
        )
        config_file = (
            static_path.joinpath("ppg-phn-model.yml")
            if config_file is None
            else Path(config_file)
        )

        static_state_dict = torch.load(model_file, map_location="cpu")
        static_state_dict = OrderedDict(
            (k, v)
            for k, v in static_state_dict.items()
            if k.startswith("base_model.encoder.") or k.startswith("phoneme_regressor")
        )

        model = cls.from_config(config_file, **kwargs)
        model.load_state_dict(static_state_dict, strict=False)
        model.eval()
        model.requires_grad_(False)
        
        return model

    @classmethod
    def from_config(
        cls,
        config: typing.Union[str, Path, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> "PPGPhonemeModel":
        if isinstance(config, (str, Path)):
            config = Path(config)
            yaml = YAML(typ="safe")
            with config.open() as fp:
                config = yaml.load(fp)

            return cls.from_config(config)

        config |= kwargs

        base_model_config = config.pop("base_model")
        base_model = PPGModel.from_config(base_model_config)
        
        return cls(base_model=base_model, **config)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        print("## 1DEBUG OUT:", speech.shape)
        feat = self.base_model(speech)
        print("## 1DEBUG OUT:", feat.shape)
        phn_prob = self.phoneme_regressor(feat)

        return phn_prob
