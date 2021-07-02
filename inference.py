from lightning_model import Wavegrad2
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import matplotlib.pyplot as plt
from utils.stft import STFTMag
import numpy as np
from g2p_en import G2p
import re

from dataloader import TextAudioDataset


def save_stft_mag(wav, fname):
    fig = plt.figure(figsize=(9, 3))
    plt.imshow(rosa.amplitude_to_db(stft(wav[0].detach().cpu()).numpy(),
               ref=np.max, top_db = 80.),
               aspect='auto',
               origin='lower',
               interpolation='none')
    plt.colorbar()
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()
    fig.savefig(fname, format='png')
    plt.close()
    return

def preprocess_eng(hparams, text):
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    print('g2p: ', phones)

    trainset = TextAudioDataset(hparams, hparams.data.train_dir, hparams.data.train_meta, train=False)

    text = trainset.get_text(phones)
    text = text.unsqueeze(0)
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Checkpoint path")
    parser.add_argument('--text',
                        type=str,
                        default=None,
                        help="raw text to synthesize, for single-sentence mode only")
    parser.add_argument('--speaker',
                        type=str,
                        default='LJSpeech',
                        help="speaker name")
    parser.add_argument('--steps',
                        type=int,
                        required=False,
                        help="Steps for sampling")
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device, 'cuda' or 'cpu'")

    args = parser.parse_args()
    #torch.backends.cudnn.benchmark = False
    hparams = OC.load('hparameter.yaml')
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)
    if args.steps is not None:
        hparams.ddpm.max_step = args.steps
        hparams.ddpm.noise_schedule = \
                "torch.tensor([1e-6,2e-6,1e-5,1e-4,1e-3,1e-2,1e-1,9e-1])"
    else:
        args.steps = hparams.ddpm.max_step
    model = Wavegrad2(hparams).to(args.device)
    stft = STFTMag()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in args.checkpoint) else ckpt)
    if hparams.data.lang == 'eng':
        text = preprocess_eng(hparams, args.text)

    speaker_dict = {spk: idx for idx, spk in enumerate(hparams.data.speakers)}
    spk_id = [speaker_dict[args.speaker]]
    spk_id = torch.LongTensor(spk_id)

    text = text.cuda()
    spk_id = spk_id.cuda()

    wav_recon, align = model.inference(text, spk_id, pace=1.1)

    save_stft_mag(wav_recon, os.path.join(hparams.log.test_result_dir, f'{args.text}.png'))
    swrite(os.path.join(hparams.log.test_result_dir, f'{args.text}.wav'),
           hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())

