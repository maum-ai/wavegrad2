import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_upsampling import GetDuration, GetRange, Upsampling


class Resampling(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.get_duration = GetDuration(hparams.encoder.channel + hparams.encoder.speaker_emb, hparams.dur_predictor.dur_lstm_channel)
        self.get_range = GetRange(hparams.encoder.channel + hparams.encoder.speaker_emb + 1, hparams.dur_predictor.range_lstm_channel)
        self.upsampling_layer = Upsampling()

    def forward(self, memory, target_duration, memory_lengths, output_lengths, no_mask=False):
        mask = None if no_mask else ~self.get_mask_from_lengths(memory_lengths)

        duration_s = self.get_duration(memory, mask, memory_lengths) # [B, N]
        sigma = self.get_range(memory, target_duration, mask, memory_lengths)  # [B, N]

        upsampled, alignments = self.upsampling_layer(memory, target_duration, sigma, output_lengths, mask)
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]

        return upsampled.transpose(1,2), alignments, duration_s, mask

    def inference(self, memory, pace):
        duration_s = self.get_duration.inference(memory) # [B, N]
        duration_s = duration_s * pace
        duration = torch.round(duration_s * (self.hparams.audio.sampling_rate / self.hparams.window.scale))
        sigma = self.get_range.inference(memory, duration)  # [B, N]

        output_len = int(torch.sum(duration, dim=-1).detach())

        upsampled, alignments = self.upsampling_layer(memory, duration, sigma, torch.tensor([output_len]), mask=None)
        # upsampled: [B, T, (chn.encoder + chn.speaker)], alignments: [B, N, T]

        return upsampled.transpose(1,2), alignments

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids < lengths.unsqueeze(1))
        return mask
