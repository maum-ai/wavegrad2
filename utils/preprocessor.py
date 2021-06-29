import os
import random
import json

import tgt
import librosa
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import wavfile

from modules.mel import Audio2Mel


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        self.audio2mel = Audio2Mel(
            filter_length=config["preprocessing"]["stft"]["filter_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
            win_length=config["preprocessing"]["stft"]["win_length"],
            sampling_rate=config["preprocessing"]["audio"]["sampling_rate"],
            n_mel_channels=config["preprocessing"]["mel"]["n_mel_channels"],
            mel_fmin=config["preprocessing"]["mel"]["mel_fmin"],
            mel_fmax=config["preprocessing"]["mel"]["mel_fmax"]
        )

    def build_from_path(self):
        print("Processing Data ...")

        # Compute duration, and mel-spectrogram
        speakers = {}
        for cv in range(2):
            out = list()
            n_frames = 0
            if cv == 0:
                dir = os.path.join(self.in_dir, 'train')
            elif cv == 1:
                dir = os.path.join(self.in_dir, 'val')
            for i, speaker in enumerate(tqdm(os.listdir(dir))):
                speakers[speaker] = i
                for wav_name in os.listdir(os.path.join(dir, speaker)):
                    if ".wav" not in wav_name or ".predict" in wav_name:
                        continue

                    basename = wav_name.replace(".wav", "")
                    tg_path_1 = os.path.join(
                        self.in_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                    )
                    tg_path_2 = os.path.join(
                        self.in_dir, "TextGrid", speaker, speaker + "_{}.TextGrid".format(basename)
                    )
                    if os.path.exists(tg_path_1):
                        ret = self.process_utterance(speaker, basename, tg_path_1, cv)
                        if ret is None:
                            continue
                        else:
                            info, n = ret
                        out.append(info)
                    elif os.path.exists(tg_path_2):
                        ret = self.process_utterance(speaker, basename, tg_path_2, cv)
                        if ret is None:
                            continue
                        else:
                            info, n = ret
                        out.append(info)

                    n_frames += n

            if cv == 0:
                # Save files
                with open(os.path.join(self.out_dir, "train", "speakers.json"), "w") as f:
                    f.write(json.dumps(speakers))

                print(
                    "Total time for train: {} hours".format(
                        n_frames * self.hop_length / self.sampling_rate / 3600
                    )
                )

                out = [r for r in out if r is not None]

                # Write metadata
                with open(os.path.join(self.out_dir, "train", "train.txt"), "w", encoding="utf-8") as f:
                    for m in out:
                        f.write(m + "\n")
            elif cv == 1:
                # Save files
                with open(os.path.join(self.out_dir, "val", "speakers.json"), "w") as f:
                    f.write(json.dumps(speakers))

                print(
                    "Total time for validation: {} hours".format(
                        n_frames * self.hop_length / self.sampling_rate / 3600
                    )
                )

                out = [r for r in out if r is not None]

                # Write metadata
                with open(os.path.join(self.out_dir, "val", "val.txt"), "w", encoding="utf-8") as f:
                    for m in out:
                        f.write(m + "\n")

    def process_utterance(self, speaker, basename, tg_path, cv):  # cv=0: train, cv=1: val
        if cv == 0:
            wav_path = os.path.join(self.in_dir, "train", speaker, "{}.wav".format(basename))
            text_path = os.path.join(self.in_dir, "train", speaker, "{}.lab".format(basename))
            os.makedirs((os.path.join(self.out_dir, "train", speaker)), exist_ok=True)
        elif cv == 1:
            wav_path = os.path.join(self.in_dir, "val", speaker, "{}.wav".format(basename))
            text_path = os.path.join(self.in_dir, "val", speaker, "{}.lab".format(basename))
            os.makedirs((os.path.join(self.out_dir, "val", speaker)), exist_ok=True)

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)
        wavfile.write(wav_path + '.cut', self.sampling_rate, wav)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute mel-scale spectrogram
        wav = torch.from_numpy(wav).view(1, 1, -1)
        mel_spectrogram = self.audio2mel(wav).squeeze(0)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        duration = torch.FloatTensor(duration)

        # Save files
        dur_filename = "duration-{}.pt".format(basename)
        mel_filename = "mel-{}.pt".format(basename)

        if cv == 0:
            torch.save(duration, os.path.join(self.out_dir, "train", speaker, dur_filename))
            torch.save(
                mel_spectrogram,
                os.path.join(self.out_dir, "train", speaker, mel_filename),
            )
        elif cv == 1:
            torch.save(duration, os.path.join(self.out_dir, "val", speaker, dur_filename))
            torch.save(
                mel_spectrogram,
                os.path.join(self.out_dir, "val", speaker, mel_filename),
            )

        return (
            "|".join([basename, speaker, text, raw_text]),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time
