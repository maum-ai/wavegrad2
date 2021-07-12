import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule


class Window(BaseModule):
    def __init__(self, hparams):
        super(Window, self).__init__()
        self.hparams = hparams
        self.scale = hparams.window.scale
        self.length = hparams.window.length

    def forward(self, y_clean, hidden_rep, output_lengths):
        y_clean_sliced = list()
        hidden_rep_sliced = list()
        for i in range(y_clean.size(0)):
            if output_lengths[i] > self.length:
                start_index = torch.randint(0, output_lengths[i] - self.length, (1,)).squeeze(0)
            else:
                start_index = 0
            hidden_rep_sliced.append(hidden_rep[i, :, start_index:start_index +
                                            self.length])
            y_clean_sliced.append(y_clean[i, self.scale * start_index:self.scale * (start_index + self.length)])
        y_clean_sliced = torch.stack(y_clean_sliced, dim=0)
        hidden_rep_sliced = torch.stack(hidden_rep_sliced, dim=0)
        if hidden_rep_sliced.size(-1) < self.length:
            hidden_rep_sliced = F.pad(hidden_rep_sliced, (0, self.length - hidden_rep_sliced.size(-1)),
                                      mode='constant', value=0.0)
            y_clean_sliced = F.pad(y_clean_sliced, (0, self.scale * self.length - y_clean_sliced.size(-1)),
                                      mode='constant', value=0.0)
        return y_clean_sliced, hidden_rep_sliced
