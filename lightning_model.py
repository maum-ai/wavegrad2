import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.io.wavfile import write as swrite
import random
import numpy as np
from omegaconf import OmegaConf

from text import Language
import dataloader
from utils.tblogger import TensorBoardLoggerExpanded
from model.encoder import TextEncoder
from model.resampling import Resampling
from model.nn import WaveGradNN
from model.window import Window


class Wavegrad2(pl.LightningModule):
    def __init__(self, hparams, train=True):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.scale = hparams.window.scale
        self.symbols = Language(hparams.data.lang, hparams.data.text_cleaners).get_symbols()
        self.symbols = ['"{}"'.format(symbol) for symbol in self.symbols]
        self.encoder = TextEncoder(hparams.chn.encoder, hparams.ker.encoder, hparams.depth.encoder, len(self.symbols))
        self.speaker_embedding = nn.Embedding(len(hparams.data.speakers), hparams.chn.speaker)
        self.resampling = Resampling(hparams)
        self.warm_start = False

        self.window = Window(hparams)
        self.decoder = WaveGradNN(hparams)
        self.filter_ratio = [1. / hparams.audio.ratio]
        self.norm = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.set_noise_schedule(hparams, train)

    # DDPM backbone is adopted form https://github.com/ivanvovk/WaveGrad
    def set_noise_schedule(self, hparams, train=True):
        self.max_step = hparams.ddpm.max_step if train \
            else hparams.ddpm.infer_step
        noise_schedule = eval(hparams.ddpm.noise_schedule) if train \
            else eval(hparams.ddpm.infer_schedule)

        self.register_buffer('betas', noise_schedule, False)
        self.register_buffer('alphas', 1 - self.betas, False)
        self.register_buffer('alphas_cumprod', self.alphas.cumprod(dim=0),
                             False)
        self.register_buffer(
            'alphas_cumprod_prev',
            torch.cat([torch.FloatTensor([1.]), self.alphas_cumprod[:-1]]),
            False)
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.]), self.alphas_cumprod])
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             alphas_cumprod_prev_with_last.sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod', self.alphas_cumprod.sqrt(),
                             False)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             (1. / self.alphas_cumprod).sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod_m1',
                             (1. - self.alphas_cumprod).sqrt() *
                             self.sqrt_recip_alphas_cumprod, False)
        posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) \
                             / (1 - self.alphas_cumprod)
        posterior_variance = torch.stack(
            [posterior_variance,
             torch.FloatTensor([1e-20] * self.max_step)])
        posterior_log_variance_clipped = posterior_variance.max(
            dim=0).values.log()
        posterior_mean_coef1 = self.betas * self.alphas_cumprod_prev.sqrt() / (
                1 - self.alphas_cumprod)
        posterior_mean_coef2 = (1 - self.alphas_cumprod_prev
                                ) * self.alphas.sqrt() / (1 -
                                                          self.alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_log_variance_clipped, False)
        self.register_buffer('posterior_mean_coef1',
                             posterior_mean_coef1, False)
        self.register_buffer('posterior_mean_coef2',
                             posterior_mean_coef2, False)

    def sample_continuous_noise_level(self, step):
        rand = torch.rand_like(step, dtype=torch.float, device=step.device)
        continuous_sqrt_alpha_cumprod = \
            self.sqrt_alphas_cumprod_prev[step - 1] * rand \
            + self.sqrt_alphas_cumprod_prev[step] * (1. - rand)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)

    def q_sample(self, y_0, step=None, noise_level=None, eps=None):
        batch_size = y_0.shape[0]
        if noise_level is not None:
            continuous_sqrt_alpha_cumprod = noise_level
        elif step is not None:
            continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[step]
        assert (step is not None or noise_level is not None)
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0, device=y_0.device)
        outputs = continuous_sqrt_alpha_cumprod * y_0 + (
                1. - continuous_sqrt_alpha_cumprod ** 2).sqrt() * eps
        return outputs

    def q_posterior(self, y_0, y, step):
        posterior_mean = self.posterior_mean_coef1[step] * y_0 \
                         + self.posterior_mean_coef2[step] * y
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[step]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def predict_start_from_noise(self, y, t, eps):
        return self.sqrt_recip_alphas_cumprod[t].unsqueeze(
            -1) * y - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps

    # t: interger not tensor
    @torch.no_grad()
    def p_mean_variance(self, y, hidden_rep, t, clip_denoised: bool):
        batch_size = y.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(
            batch_size, 1)
        eps_recon = self.decoder(y, hidden_rep, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def compute_inverse_dynamincs(self, y, hidden_rep, t, clip_denoised=False):
        model_mean, model_log_variance = self.p_mean_variance(
            y, hidden_rep, t, clip_denoised)
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, hidden_rep,
               start_step=None,
               store_intermediate_states=False):
        batch_size, T = hidden_rep.shape[0], hidden_rep.shape[-1]
        start_step = self.max_step if start_step is None \
            else min(start_step, self.max_step)
        step = torch.tensor([start_step] * batch_size,
                            dtype=torch.long,
                            device=self.device)
        y_t = torch.randn(batch_size, T * self.scale, device=self.device)
        ys = [y_t]
        t = start_step - 1
        while t >= 0:
            y_t = self.compute_inverse_dynamincs(y_t, hidden_rep, t)
            ys.append(y_t)
            t -= 1
        return ys if store_intermediate_states else ys[-1]

    def forward(self, text, wav, duration_target, speakers, input_lengths, output_lengths, noise_level, no_mask=False):
        text_encoding = self.encoder(text, input_lengths)  # [B, N, chn.encoder]
        speaker_emb = self.speaker_embedding(speakers)  # [B, chn.speaker]
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, text_encoding.size(1), -1)  # [B, N, chn.speaker]
        decoder_input = torch.cat((text_encoding, speaker_emb), dim=2)
        hidden_rep, alignment, duration, mask = \
            self.resampling(decoder_input, duration_target, input_lengths, output_lengths, no_mask)
        wav_sliced, hidden_rep_sliced = self.window(wav, hidden_rep, output_lengths)
        eps = torch.randn_like(wav_sliced, device=wav.device)
        wav_noisy_sliced = self.q_sample(wav_sliced, noise_level=noise_level, eps=eps)
        eps_recon = self.decoder(wav_noisy_sliced, hidden_rep_sliced, noise_level)
        return eps_recon, eps, wav_sliced, wav_noisy_sliced, hidden_rep_sliced, alignment, duration, mask

    def common_step(self, text, wav, duration_target, speakers, input_lengths, output_lengths, step, no_mask=False):
        noise_level = self.sample_continuous_noise_level(step) \
            if self.training \
            else self.sqrt_alphas_cumprod_prev[step].unsqueeze(-1)
        eps_recon, eps, wav_sliced, wav_noisy_sliced, hidden_rep_sliced, alignment, duration, mask = \
            self(text, wav, duration_target, speakers, input_lengths, output_lengths, noise_level)
        noise_loss = self.norm(eps_recon, eps)

        mask = ~mask
        duration = duration.masked_select(mask)
        duration_target = duration_target.masked_select(mask)
        duration_loss = self.mse_loss(duration, duration_target / (self.hparams.audio.sampling_rate / self.hparams.window.scale))

        loss = noise_loss + self.hparams.train.loss_rate.dur * duration_loss
        return loss, noise_loss, duration_loss, wav_sliced, wav_noisy_sliced, eps, eps_recon, hidden_rep_sliced, alignment

    def inference(self, text, speakers, max_decoder_steps, mel_chunk_size, prenet_dropout=0.5, attn_reset=False, pace=1.0):
        text_encoding = self.encoder.inference(text)
        speaker_emb = self.speaker_embedding(speakers)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
        if text_encoding.dtype == torch.float16:
            speaker_emb = speaker_emb.half()
        decoder_input = torch.cat((text_encoding, speaker_emb), dim=2)
        hidden_rep, _ = self.resampling(decoder_input, pace=pace)
        wav_recon = self.sample(hidden_rep, store_intermediate_states=False)
        return wav_recon

    def training_step(self, batch, batch_idx):
        text, wav, duration_target, speakers, input_lengths, output_lengths, max_input_len = batch
        step = torch.randint(
            0, self.max_step, (wav.shape[0],), device=self.device) + 1
        loss, noise_loss, duration_loss, *_ = \
            self.common_step(text, wav, duration_target, speakers, input_lengths, output_lengths, step)

        self.log('train/noise_loss', noise_loss, sync_dist=True)
        self.log('train/duration_loss', duration_loss, sync_dist=True)
        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        text, wav, duration_target, speakers, input_lengths, output_lengths, max_input_len = batch

        step = torch.randint(
            0, self.max_step, (wav.shape[0],), device=self.device) + 1
        loss, noise_loss, duration_loss, wav, wav_noisy, eps, eps_recon, hidden_rep, alignment = \
            self.common_step(text, wav, duration_target, speakers, input_lengths, output_lengths, step)

        self.log('val/noise_loss', noise_loss, sync_dist=True)
        self.log('val/duration_loss', duration_loss, sync_dist=True)
        self.log('val/loss', loss, sync_dist=True)
        if batch_idx == 0:
            i = torch.randint(0, wav.shape[0], (1,)).item()
            wav_recon = self.predict_start_from_noise(wav_noisy, step - 1,
                                                    eps_recon)
            eps_error = eps - eps_recon
            hidden_rep_i = hidden_rep[i].unsqueeze(0)
            wav_recon_allstep = self.sample(hidden_rep_i, store_intermediate_states=False)
            self.trainer.logger.log_spectrogram(wav[i], wav_noisy[i],
                                                wav_recon[i], eps_error[i], wav_recon_allstep[0],
                                                step[i].item(),
                                                self.current_epoch)
            self.trainer.logger.log_audio(wav[i], wav_noisy[i],
                                          wav_recon[i], wav_recon_allstep[0], self.current_epoch)
            self.trainer.logger.log_alignment(alignment[i], self.current_epoch)

        return {
            'val_loss': loss,
        }

    def configure_optimizers(self):
        if self.warm_start:
            learnable_params = self.speaker_embedding.parameters()
        else:
            learnable_params = self.parameters()
        return torch.optim.Adam(
            learnable_params,
            lr=self.hparams.train.adam.lr,
            weight_decay=self.hparams.train.adam.weight_decay,
        )

    # def lr_lambda(self, step):
    #     progress = (step - self.hparams.train.decay.start) / (self.hparams.train.decay.end - self.hparams.train.decay.start)
    #     return self.hparams.train.decay.rate ** np.clip(progress, 0.0, 1.0)
    #
    # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_idx, second_order_closure, using_native_amp, using_lbfgs, on_tpu=False):
    #     lr_scale = self.lr_lambda(self.global_step)
    #     for pg in optimizer.param_groups:
    #         pg['lr'] = lr_scale * self.hparams.train.adam.lr
    #
    #     optimizer.step()
    #     optimizer.zero_grad()
    #
    #     self.trainer.logger.log_learning_rate(lr_scale * self.hparams.train.adam.lr, self.global_step)

    def train_dataloader(self):
        return dataloader.create_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_dataloader(self.hparams, 1)
