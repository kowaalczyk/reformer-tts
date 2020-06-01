import matplotlib.pyplot as plt
from torch import Tensor

from reformer_tts.dataset.convert import spectrogram_to_log_scale


def plot_spectrogram(
        mel_spectrogram: Tensor,
        scale: bool = False,
        max_len: int = 400
) -> plt.Axes:
    """
    Plot is clipped to max_len samples in order to have reasonable width.

    Usage:
    plot_spectrogram(x)
    plt.savefig(filename.png) # save to file
    plt.show() # display in jupyer

    Or:
    ax = plot_spectrogram(x)
    ... do something with ax ...
    """
    fig_width = 8. * max_len / 100  # height is always 100
    plt.figure(figsize=(fig_width, 8.))

    if scale:
        formatted_spectrogram = spectrogram_to_log_scale(mel_spectrogram)
    else:
        formatted_spectrogram = mel_spectrogram

    cutoff = min(formatted_spectrogram.shape[-1], max_len)
    formatted_spectrogram = formatted_spectrogram[0, :, :cutoff]
    ax = plt.imshow(formatted_spectrogram.numpy(), cmap='gray')
    return ax


def plot_attention_matrix(attention_matrix: Tensor) -> plt.Figure:
    assert attention_matrix.dim() == 2
    aspect = attention_matrix.shape[1] / attention_matrix.shape[0]
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.matshow(attention_matrix.numpy(), aspect=aspect)
    return figure
