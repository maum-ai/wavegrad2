import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule


class Window(BaseModule):
    def __init__(self, hparams):
        super(Wndow, self).__init__()
        self.hparams = hparams
        self.scale = hparams.window.scale
        self.length = hparams.window.length

    def forward(self, yn, hidden_rep, output_length):
        start_index = (torch.rand(output_length.size(), device=yn.device) *
                       (output_length - self.length)).long()
        yn_sliced = list()
        hidden_sliced = list()
        for i in range(yn.size(0)):
            hidden_sliced.append(hidden_rep[i, :, start_index:start_index +
                                            self.length])
            yn_sliced.append(yn[i, :, self.scale * start_index:self.scale *
                                (start_index + self.length)])
        yn_sliced = torch.stack(yn_sliced, dim=0)
        hidden_sliced = torch.stack(hidden_sliced, dim=0)
        return yn_sliced, hidden_sliced
