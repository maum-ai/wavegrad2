import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_upsampling import DurationPredictor, RangeParameterPredictor, GaussianUpsampling


class Resampling(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.duration_predictor = DurationPredictor(hparams.encoder.channel + hparams.encoder.speaker_emb, hparams.dur_predictor.dur_lstm_channel)
        self.range_parameter_predictor = RangeParameterPredictor(hparams.encoder.channel + hparams.encoder.speaker_emb + 1, hparams.dur_predictor.range_lstm_channel)
        self.gaussian_upsampling = GaussianUpsampling()

    def forward(self, memory, target_duration, memory_lengths, output_lengths, no_mask=False):
        mask = None if no_mask else ~self.get_mask_from_lengths(memory_lengths)

        duration_s = self.duration_predictor(memory, mask, memory_lengths) # [B, N]
        sigma = self.range_parameter_predictor(memory, target_duration, mask, memory_lengths)  # [B, N]

        upsampled, alignments = self.gaussian_upsampling(memory, target_duration, sigma, output_lengths, mask)
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]

        return upsampled.transpose(1,2), alignments, duration_s, mask

    def inference(self, memory, pace):
        duration_s = self.duration_predictor.inference(memory) # [B, N]
        duration_s = duration_s * pace
        duration = torch.round(duration_s * (self.hparams.audio.sampling_rate / self.hparams.window.scale))
        sigma = self.range_parameter_predictor.inference(memory, duration)  # [B, N]

        output_len = int(torch.sum(duration, dim=-1).detach())

        upsampled, alignments = self.gaussian_upsampling(memory, duration, sigma, torch.tensor([output_len]), mask=None)
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]

        return upsampled.transpose(1,2), alignments, duration, sigma

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids < lengths.unsqueeze(1))
        return mask
