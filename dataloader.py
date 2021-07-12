#This code is adopted from
#https://github.com/NVIDIA/tacotron2
import os
import re
import torch
import random
import librosa
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter

from text import Language
from text.cmudict import CMUDict


class TextAudioDataset(Dataset):
    def __init__(self, hparams, data_dir, metadata_path, train=True):
        super().__init__()
        self.hparams = hparams
        self.lang = Language(hparams.data.lang, hparams.data.text_cleaners)
        self.train = train
        self.data_dir = data_dir
        metadata_path = os.path.join(data_dir, metadata_path)
        self.meta = self.load_metadata(metadata_path)
        self.speaker_dict = {speaker: idx for idx, speaker in enumerate(hparams.data.speakers)}

        if train:
            # balanced sampling for each speaker
            speaker_counter = Counter((spk_id \
                                       for basename, spk_id, text, raw_text in self.meta))
            weights = [1.0 / speaker_counter[spk_id] \
                       for basename, spk_id, text, raw_text in self.meta]

            self.mapping_weights = torch.DoubleTensor(weights)

        if hparams.data.lang == 'eng2':
            self.cmudict = CMUDict(hparams.data.cmudict_path)
            self.cmu_pattern = re.compile(r'^(?P<word>[^!\'(),-.:~?]+)(?P<punc>[!\'(),-.:~?]+)$')

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            idx = torch.multinomial(self.mapping_weights, 1).item()

        basename, spk_id, text, raw_text = self.meta[idx]
        audio_path = os.path.join(self.data_dir, 'wav', '{}-wav-{}.wav'.format(spk_id, basename))
        wav, _ = librosa.load(audio_path, self.hparams.audio.sample_rate)
        wav = torch.from_numpy(wav)
        text_norm = self.get_text(text)
        dur_path = os.path.join(self.data_dir, 'duration', '{}-duration-{}.npy'.format(spk_id, basename))
        duration = np.load(dur_path)
        duration = torch.from_numpy(duration)
        spk_id = self.speaker_dict[spk_id]
        return text_norm, wav, duration.float(), spk_id

    def get_text(self, text):
        # if lang='eng2', then use representation mixing. (arXiv:1811.07240)
        # i.e., randomly apply CMUDict-based English g2p to whole sentence.
        # note that lang='eng2' will use arpabet WITH stress.
        if self.hparams.data.lang == 'eng2' and random.random() < 0.5:
            text = ' '.join([self.get_arpabet(word) for word in text.split(' ')])
        text_norm = torch.LongTensor(self.lang.text_to_sequence(text, self.hparams.data.text_cleaners))
        return text_norm

    def get_arpabet(self, word):
        arpabet = self.cmudict.lookup(word)
        if arpabet is None:
            match = self.cmu_pattern.search(word)
            if match is None:
                return word
            subword = match.group('word')
            arpabet = self.cmudict.lookup(subword)
            if arpabet is None:
                return word
            punc = match.group('punc')
            arpabet = '{%s}%s' % (arpabet[0], punc)
        else:
            arpabet = '{%s}' % arpabet[0]

        if random.random() < 0.5:
            return word
        else:
            return arpabet

    def load_metadata(self, path, split="|"):
        metadata = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip().split(split)
                if self.hparams.train.fine_tuning and stripped[2] != self.hparams.train.tuning_speaker:
                    continue
                metadata.append(stripped)

        return metadata


def create_dataloader(hparams, cv):
    def collate_fn(batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = torch.empty(len(batch), dtype=torch.long)
        max_input_len.fill_(input_lengths[0])

        text_padded = torch.zeros((len(batch), max_input_len[0]), dtype=torch.long)
        max_target_len = max([x[1].size(0) for x in batch])
        if max_target_len % hparams.window.scale != 0:
            max_target_len = max_target_len + (hparams.window.scale - max_target_len % hparams.window.scale)

        wav_padded = torch.zeros(len(batch), max_target_len)
        output_lengths = torch.empty(len(batch), dtype=torch.long)
        speakers = torch.empty(len(batch), dtype=torch.long)

        duration_padded = torch.zeros((len(batch), max_input_len[0]), dtype=torch.long)

        for idx, key in enumerate(ids_sorted_decreasing):
            text = batch[key][0]
            text_padded[idx, :text.size(0)] = text
            wav = batch[key][1]
            wav_padded[idx, :wav.size(0)] = wav
            duration = batch[key][2]
            duration_padded[idx, :duration.size(0)] = duration
            output_lengths[idx] = wav.size(0) // hparams.window.scale
            speakers[idx] = batch[key][3]

        return text_padded, wav_padded, duration_padded, speakers, \
               input_lengths, output_lengths, max_input_len
    if cv == 0:
        trainset = TextAudioDataset(hparams, hparams.data.train_dir, hparams.data.train_meta, train=True)
        return DataLoader(trainset, batch_size=hparams.train.batch_size, shuffle=True,
                          num_workers=hparams.train.num_workers,
                          collate_fn=collate_fn, pin_memory=True, drop_last=False)
    else:
        valset = TextAudioDataset(hparams, hparams.data.val_dir, hparams.data.val_meta, train=False)
        return DataLoader(valset, batch_size=hparams.train.batch_size, shuffle=False,
                          num_workers=hparams.train.num_workers,
                          collate_fn=collate_fn, pin_memory=False, drop_last=False)