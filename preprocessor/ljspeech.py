#This code is adopted from
#https://github.com/ming024/FastSpeech2
import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import Language


def prepare_align(hparams):
    in_dir = hparams.path.corpus_path
    out_dir = hparams.path.raw_path
    sampling_rate = hparams.preprocessing.audio.sampling_rate
    max_wav_value = hparams.preprocessing.audio.max_wav_value
    cleaners = hparams.preprocessing.text.text_cleaners
    language = Language(hparams.preprocessing.text.lang, cleaners)
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = language._clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
