from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch.nn.functional as F
from os import path, makedirs
from omegaconf import OmegaConf as OC
from datetime import datetime, timedelta
from utils.stft import STFTMag
import librosa as rosa


class TensorBoardLoggerExpanded(TensorBoardLogger):
    def __init__(self, hparam):
        super().__init__(hparam.log.tensorboard_dir, name=hparam.name,
                default_hp_metric= False)
        self.hparam = hparam
        self.log_hyperparams(hparam)

        self.stftmag = STFTMag()

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        return data

    def plot_spectrogram_to_numpy(self, y, y_noisy, y_recon,
                                  eps_error, y_recon_allstep, alignment, step):
        fig = plt.figure(figsize=(9, 18))

        ax = plt.subplot(6, 1, 1)
        ax.set_title('alignment')
        plt.imshow(alignment, aspect='auto', origin='lower', interpolation='none',
                   norm=Normalize(vmin=0.0, vmax=1.0))
        plt.colorbar()
        plt.xlabel('Hidden representation timestep')
        plt.ylabel('Encoder timestep')
        plt.tight_layout()

        name_list = ['y_recon_allstep', 'y', f'y_noisy(Diffstep_{step})', 'y_recon', 'error_recon']
        for i, yy in enumerate([y_recon_allstep, y, y_noisy, y_recon, eps_error]):
            ax=plt.subplot(6, 1, i + 2)
            ax.set_title(name_list[i])
            plt.imshow(rosa.amplitude_to_db(self.stftmag(yy).numpy(),
                       ref=np.max,top_db=80.),
                       #vmin = -20,
                       vmax = 0.,
                       aspect='auto',
                       origin='lower',
                       interpolation='none')
            plt.colorbar()
            plt.xlabel('Frames')
            plt.ylabel('Channels')
            plt.tight_layout()

        fig.canvas.draw()
        data = self.fig2np(fig)

        plt.close()
        return data

    @rank_zero_only
    def log_spectrogram(self, y, y_noisy, y_recon, eps_error, y_recon_allstep, alignment,
                        diff_step, epoch):
        y, y_noisy, y_recon, eps_error, y_recon_allstep, alignment = \
            y.detach().cpu(), y_noisy.detach().cpu(), y_recon.detach().cpu(), \
            eps_error.detach().cpu(), y_recon_allstep.detach().cpu(), alignment.detach().cpu()
        spec_img = self.plot_spectrogram_to_numpy(
                y, y_noisy, y_recon,
                eps_error, y_recon_allstep, alignment, diff_step)
        self.experiment.add_image(path.join(self.save_dir, 'result'),
                                  spec_img,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
        return

    @rank_zero_only
    def log_audio(self, y, y_noisy, y_recon, y_recon_allstep, epoch):
        y, y_noisy, y_recon, y_recon_allstep = y.detach().cpu(), y_noisy.detach().cpu(), \
                                               y_recon.detach().cpu(), y_recon_allstep.detach().cpu()

        name_list = ['y', 'y_noisy', 'y_recon', 'y_recon_allstep']

        for n, yy in zip(name_list, [y, y_noisy, y_recon, y_recon_allstep]):
            self.experiment.add_audio(n,
                                      yy, epoch, self.hparam.audio.sampling_rate)

        self.experiment.flush()
        return

    @rank_zero_only
    def log_learning_rate(self, learning_rate, step):
        self.experiment.add_scalar('learning_rate', learning_rate, step)
    
