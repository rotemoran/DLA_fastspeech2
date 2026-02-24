import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()

        # Duration & length
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.energy_predictor = VariancePredictor(model_config)

        # Pitch predictor replaced with 1D conv + projection
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        hidden_size = model_config["transformer"]["encoder_hidden"]
        dropout = model_config["variance_embedding"].get("dropout", 0.1)
        num_scales = model_config["variance_embedding"].get("num_scales", 10)  # number of CWT scales

        # Energy bins / embedding remain the same
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            energy_min, energy_max = stats["energy"][:2]

        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )
        self.energy_embedding = nn.Embedding(n_bins, hidden_size)

        # ------------------------
        # PITCH PREDICTOR NETWORK
        # ------------------------
        # 2-layer 1D conv with ReLU + layernorm + dropout
        self.pitch_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            Transpose(1, 2),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            Transpose(1, 2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            Transpose(1, 2),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            Transpose(1, 2),
        )

        # Project to pitch CWT spectrogram (num_scales)
        self.pitch_proj = nn.Linear(hidden_size, num_scales)

        # iCWT weights per paper Appendix C Eq 2: F̂_0(t) = Σ_i Ŵ_i(t)(i+2.5)^{-5/2}
        icwt_weights = np.array(
            [(i + 2.5) ** (-5.0 / 2.0) for i in range(1, num_scales + 1)],
            dtype=np.float32,
        )
        self.register_buffer("icwt_weights", torch.from_numpy(icwt_weights).view(1, 1, -1))

        # Predict mean/variance of original utterance-level pitch contour
        self.pitch_stat_proj = nn.Linear(hidden_size, 2)  # outputs [mean, variance]

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
    self,
    x,
    src_mask,
    mel_mask=None,
    max_len=None,
    pitch_target=None,
    energy_target=None,
    duration_target=None,
    p_control=1.0,
    e_control=1.0,
    d_control=1.0,
    ):
        # ------------------------
        # Duration
        # ------------------------
        log_duration_prediction = self.duration_predictor(x, src_mask)

        # ------------------------
        # Energy
        # ------------------------
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        # ------------------------
        # Length regulation
        # ------------------------
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        # ------------------------
        # Pitch predictor (CWT)
        # ------------------------
        # x: (B, T, hidden)
        x_conv = self.pitch_conv(x.transpose(1, 2))      # (B, hidden, T)
        x_conv = x_conv.transpose(1, 2)                  # (B, T, hidden)

        pitch_prediction = self.pitch_proj(x_conv)       # (B, T, num_scales)

        # Utterance-level mean/variance
        global_vec = x_conv.mean(dim=1)                  # (B, hidden)
        pitch_mean_var = self.pitch_stat_proj(global_vec)  # (B, 2)

        # iCWT: recover 1D pitch contour from CWT spectrogram (paper Appendix C Eq 2)
        pitch_1d = (pitch_prediction * self.icwt_weights).sum(dim=-1, keepdim=True)  # (B, T, 1)

        # Add pitch to encoder features (frame-level); paper uses 1D pitch after iCWT
        if self.pitch_feature_level in ["phoneme_level", "frame_level"]:
            x = x + pitch_1d
            
        # Apply prosody control scaling only during inference
        if pitch_target is None and p_control != 1.0:
            pitch_prediction = pitch_prediction * p_control

        # ------------------------
        # Energy (frame-level)
        # ------------------------
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding

        return (
            x,                  # updated hidden states
            pitch_prediction,   # CWT pitch spectrogram (B, T, num_scales)
            pitch_mean_var,     # utterance-level mean/variance (B, 2)
            energy_prediction,  # energy
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )
 


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        return x
    
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
