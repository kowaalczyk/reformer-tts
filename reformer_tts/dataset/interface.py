import random
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List, Callable

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import trange


class TextToSpectrogramDataset(Dataset):
    """ Text-to-Spectrogram interface for the Trump Speech Dataset """

    def __init__(self, merged_transcript_csv_path: Path, mel_directory: Path):
        self.transcripts_df = pd.read_csv(merged_transcript_csv_path)
        self.mel_directory = mel_directory
        phonemes = {p for s in self.transcripts_df['phonemes'] for p in s.split()}
        self.phoneme_to_idx = {p: i + 1 for i, p in enumerate(sorted(list(phonemes)))}

    def __len__(self):
        return len(self.transcripts_df)

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        :param idx: index or single-element tensor / numpy array
        :return: { phonemes: Tensor(len), spectrogram: Tensor(len x n_mels), idx:int }
        """
        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        phonemes = self.transcripts_df.loc[idx, "phonemes"]
        phoneme_indices = torch.LongTensor([self.phoneme_to_idx[p] for p in phonemes.split()])

        audio_path = Path(self.transcripts_df.loc[idx, "audio_path"])
        spectrogram_path = Path(self.mel_directory / audio_path.name) \
            .with_suffix(".pt")
        spectrogram = torch.load(spectrogram_path)
        spectrogram = spectrogram.view(spectrogram.shape[1:]).transpose(1, 0)

        sample = {"phonemes": phoneme_indices, "spectrogram": spectrogram, "idx": idx}
        return sample


class SpectrogramToSpeechDataset(Dataset):
    """ Spectrogram-to-Speech interface for the Trump Speech Dataset """

    def __init__(
            self,
            merged_transcript_csv_path: Path,
            mel_directory: Path,
            audio_segment_length: int,
            mel_hop_length: int,
    ):
        self.transcripts_df = pd.read_csv(merged_transcript_csv_path)
        self.mel_directory = mel_directory
        self.audio_segment_length = audio_segment_length
        self.mel_hop_length = mel_hop_length

    def __len__(self):
        return len(self.transcripts_df)

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        :param idx: index or single-element tensor / numpy array
        :return: { audio: Tensor(audio_segment_length), spectrogram: Tensor(n_mels x mel_len) }
        """
        audio, spectrogram = self.raw_sample(idx)

        # based on https://github.com/tianrengao/SqueezeWave/blob/177aebb5a4c53bd70100a6e8676c338582fbcd55/mel2samp.py#L90
        audio_size = audio.size(0)
        n_hops = self.audio_segment_length // self.mel_hop_length
        if audio_size >= self.audio_segment_length:
            max_audio_start = audio_size - self.audio_segment_length
            random_hop = random.randint(0, max_audio_start // self.mel_hop_length)

            audio_start = random_hop * self.mel_hop_length
            audio_end = audio_start + n_hops * self.mel_hop_length
            audio = audio[audio_start:audio_end]

            spectrogram_start = random_hop
            spectrogram_end = spectrogram_start + n_hops
            spectrogram = spectrogram[:, spectrogram_start:spectrogram_end]
        else:
            audio_pad_size = self.audio_segment_length - audio_size
            audio = torch.nn.functional.pad(audio, [0, audio_pad_size], 'constant')

            spectrogram_pad_size = n_hops - spectrogram.shape[-1]
            spectrogram = torch.nn.functional.pad(
                spectrogram,
                [0, spectrogram_pad_size],
                'constant'
            )

        sample = {"audio": audio, "spectrogram": spectrogram, "idx": idx}
        return sample

    def raw_sample(self, idx: Union[torch.Tensor, np.array, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full-length audio and spectrogram at the given index
        :return: tuple: audio: Tensor(audio_segment_length), spectrogram: Tensor(n_mels x mel_len)
        """

        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        audio_path = Path(self.transcripts_df.loc[idx, "audio_path"])
        audio, _ = torchaudio.load(audio_path)  # ignore sampling rate here
        audio = torch.squeeze(audio, dim=0)

        spectrogram_path = Path(self.mel_directory / audio_path.name) \
            .with_suffix(".pt")
        spectrogram = torch.load(spectrogram_path)
        spectrogram = torch.squeeze(spectrogram, dim=0)

        return audio, spectrogram


class VariableBatchSizeDataset(Dataset):
    """
    Wrapper for existing datasets that groups samples into batches of same size
    (in terms of memory) but variable size (in terms of number of samples).

    The dataset groups samples into buckets by length, one bucket per each power of 2,
    this way, empty padding is never more than 50% of any single batch.

    For example, if we know we can fit batch of size 8 and length 1024 in memory,
    it means that we can also fit batch with size 16 and length 512,
    size 32 and length 256 and so on.
    """

    def __init__(
            self,
            dataset: Union[TextToSpectrogramDataset, SpectrogramToSpeechDataset],
            collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
            min_mel_len: int = 64,
            max_mel_len: int = 8192,  # 1024 ~10s
            batch_size_for_max_len: int = 1,
    ):
        """
        Creates a variable batch size dataset from the provided dataset.

        The dataset has to yield items containing "spectrogram" key, based on
        which the length of samples will be determined for bucketing.

        :param dataset: underlying dataset (collection of samples)
        :param collate_fn: function applied to list(sample) that should produce a batch dict
        :param min_mel_len: smallest possible length of mel spectrogram in dataset (smaller will produce errors)
        :param max_mel_len: upper bound on mel spectrogram length (larger will be dropped)
        :param batch_size_for_max_len: maximal batch size for max_mel_len that fits in memory
        """
        assert np.log2(min_mel_len) == int(np.log2(min_mel_len)), "min_mel_len must be a power of 2"
        assert np.log2(max_mel_len) == int(np.log2(max_mel_len)), "max_mel_len must be a power of 2"

        min_bucket_key = int(np.log2(min_mel_len))
        n_buckets = int(np.log2(max_mel_len)) - min_bucket_key
        buckets = dict()  # log2(bucket_min_mel_len) => list(batch: np.array)
        buckets_queue = dict()  # log2(bucket_min_mel_len) => list(sample)
        batch_size = dict()  # log2(bucket_min_mel_len) => int
        for i in range(n_buckets):
            # min_mel_len is now minimal length of items in the i-th bucket
            # current bucket key (min_bucket_key + i) == int(log2(min_mel_len))
            buckets[min_bucket_key + i] = list()
            buckets_queue[min_bucket_key + i] = list()
            batch_size[min_bucket_key + i] = batch_size_for_max_len * 2**(n_buckets - i - 1)
            min_mel_len *= 2
        assert min_mel_len == max_mel_len

        dropped_samples = list()
        for i in trange(len(dataset), desc="bucketing samples"):
            sample = dataset[i]
            sample_len = max(sample["spectrogram"].shape)  # todo: document
            if sample_len >= max_mel_len:
                dropped_samples.append(sample["idx"])
                continue

            sample_bucket_key = int(np.log2(sample_len))
            buckets_queue[sample_bucket_key].append(sample["idx"])
            if len(buckets_queue[sample_bucket_key]) == batch_size[sample_bucket_key]:
                # queue for bucket has achieved max size, so we form a batch and add it to the bucket
                batch = np.array(buckets_queue[sample_bucket_key], dtype=np.int)
                buckets[sample_bucket_key].append(batch)
                buckets_queue[sample_bucket_key] = list()

        # incomplete buckets are also dropped
        for bucket_key in buckets_queue:
            for idx in buckets_queue[bucket_key]:
                dropped_samples.append(idx)
        del buckets_queue

        if len(dropped_samples) > 0:
            print(f"WARNING: there are {len(dropped_samples)} dropped samples")
            print(f"List is available via {self.__class__.__name__}.dropped_samples attribute")

        self.dropped_samples: List[int] = dropped_samples
        self.buckets: Dict[int, List[np.array]] = buckets
        self.bucket_keys: np.array = np.array(list(buckets.keys()), dtype=np.int)
        self.cum_bucket_lens: np.array = np.cumsum([len(buckets[k]) for k in buckets], dtype=np.int)
        self.batch_size = batch_size
        self.dataset: Dataset = dataset
        self.collate_fn = collate_fn

    def __getitem__(self, idx: Union[torch.Tensor, np.array, int]) -> Dict[str, Any]:
        """
        Get a variable-sized batch of samples (output of collate_fn called on a list of samples).

        Samples within the batch are randomly shuffled every time this function is called.
        This is necessary, as shuffling samples in the underlying raw dataset after epoch ends
        would result in data leakage from test set (or validation set) to the train set.
        """

        if hasattr(idx, '__getitem__'):
            idx = int(idx[0])

        # we rely on a hack that np.argmax returns first maximal result
        # (in our case: idx of bucket that cumulatively has more elements than idx)
        bucket_idx = np.argmax(self.cum_bucket_lens > idx)
        bucket_key = self.bucket_keys[bucket_idx]
        if bucket_idx == 0:
            batch_key_in_bucket = idx
        else:
            batch_key_in_bucket = idx - self.cum_bucket_lens[bucket_idx - 1]

        idx_batch: np.array = self.buckets[bucket_key][batch_key_in_bucket]
        np.random.shuffle(idx_batch)  # shuffle samples inside the batch in place

        batch = [self.dataset[int(idx)] for idx in idx_batch]
        batch = self.collate_fn(batch)
        return batch

    def __len__(self):
        """ Number of variable-length batches in the dataset """
        return self.cum_bucket_lens[-1]

    def info(self) -> pd.DataFrame:
        """ Get a dataframe with info about every bucket """
        dfs = []
        for bucket_key, bucket_batches in self.buckets.items():
            df = pd.DataFrame(data={
                "batch_size": [self.batch_size[bucket_key]],
                "batches": [len(bucket_batches)],
                "mel_length": [2 ** (bucket_key + 1)]
            }, index=[bucket_key])
            dfs.append(df)
        info_df = pd.concat(dfs, axis="index")
        info_df.index.name = "bucket"
        return info_df
