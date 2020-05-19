from reformer_tts.dataset.interface import *
from reformer_tts.dataset.utils import custom_sequence_padder


def test_vbs_dataset_info(config):
    dataset = TextToSpectrogramDataset(config.merged_transcript_csv_path, config.mel_directory)

    vbs_dataset = VariableBatchSizeDataset(dataset, custom_sequence_padder)
    info = vbs_dataset.info()

    assert info.loc[6, "batch_size"] == 64
    assert info.loc[12, "batch_size"] == 1
    assert info.loc[6, "mel_length"] == 128
    assert info.loc[12, "mel_length"] == 8192


def test_vbs_dataset_first_sample(config):
    dataset = TextToSpectrogramDataset(config.merged_transcript_csv_path, config.mel_directory)

    vbs_dataset = VariableBatchSizeDataset(dataset, custom_sequence_padder)
    sample = vbs_dataset[0]

    # correct keys:
    assert sample.get("spectrogram") is not None
    assert sample.get("phonemes") is not None

    # keys have the same (correct) batch size:
    assert sample["spectrogram"].shape[0] == 64
    assert sample["phonemes"].shape[0] == 64


def test_vbs_dataset_last_sample(config):
    dataset = TextToSpectrogramDataset(config.merged_transcript_csv_path, config.mel_directory)

    # we use max_mel_len to make sure all datasets have last sample in the same bucket
    vbs_dataset = VariableBatchSizeDataset(dataset, custom_sequence_padder, max_mel_len=1024)
    sample = vbs_dataset[len(vbs_dataset) - 1]

    # correct keys:
    assert sample.get("spectrogram") is not None
    assert sample.get("phonemes") is not None

    # keys have the same (correct) batch size:
    assert sample["spectrogram"].shape[0] == 1
    assert sample["phonemes"].shape[0] == 1
