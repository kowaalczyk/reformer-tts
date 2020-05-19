import torch
from torch.nn.utils.rnn import pad_sequence


def custom_sequence_padder(batch):
    """
    :param batch: Collection({ phonemes: Tensor(len), spectrogram: Tensor(len x n_mels) })
    :return: { phonemes: Tensor(batch x len),
               spectrogram: Tensor(batch x len x n_mels),
               stop_tokens: Tensor(batch x  len)
             }
    """
    phonemes = [e['phonemes'] for e in batch]
    phonemes = pad_sequence(phonemes, batch_first=True)

    spectrograms = [e['spectrogram'] for e in batch]
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    start_token = torch.zeros((spectrograms.shape[0], 1, spectrograms.shape[2]), device=torch.device('cpu'))
    spectrograms = torch.cat([start_token, spectrograms], dim=1)

    length_ind = torch.tensor([len(e['spectrogram']) for e in batch], dtype=torch.long, device=torch.device('cpu'))
    lengths_matrix = torch.zeros((spectrograms.shape[0], spectrograms.shape[1]), device=torch.device('cpu'))
    lengths_matrix[torch.arange(len(length_ind), device=torch.device('cpu')), length_ind] = 1
    stop_tokens = lengths_matrix[:, 1:]

    mask_rows = []
    max_spectrogram_length = spectrograms.shape[1] - 1
    for length in length_ind:
        ones = torch.ones(length, device=torch.device('cpu'))
        zeros = torch.zeros(max_spectrogram_length - length, device=torch.device('cpu'))
        mask_row = torch.cat([ones, zeros])
        mask_row = mask_row.unsqueeze(1).repeat(1, spectrograms.shape[2])
        mask_rows.append(mask_row)
    loss_mask = torch.stack(mask_rows)
    assert loss_mask.shape == spectrograms[:, 1:, :].shape

    return {
        "phonemes": phonemes,
        "spectrogram": spectrograms,
        "stop_tokens": stop_tokens,
        "loss_mask": loss_mask,
    }


def get_subset_lengths(length, split_percentages):
    """
    :param length: length of dataset
    :param split_percentages: triple of floats that sum to 1
    :return: triple of ints - lengths of subsets
    """
    assert sum(split_percentages) == 1, "split_percentages have to sum up to 1"
    assert len(split_percentages) == 3, "expected percentages for exactly 3 subsets"

    test_l = int(length * split_percentages[2]) if split_percentages[2] != 0 else 0
    val_l = int(length * split_percentages[1]) if split_percentages[1] != 0 else 0
    train_l = length - test_l - val_l
    return train_l, val_l, test_l
