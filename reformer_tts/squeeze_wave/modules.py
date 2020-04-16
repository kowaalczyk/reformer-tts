from dataclasses import asdict
from time import time

import torch
import torch.nn.functional as F

from reformer_tts.squeeze_wave.config import WNConfig


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    A sequence of combined transformations for speed improvement
    :param input_a: of shape (*, 2 * n_channels, L)
    :param input_b: of shape (*, 2 * n_channels, L)
    :param n_channels: IntTensor containing a single item (int: n_channels)
    :return: of shape (*, n_channels, L)
    """
    n_channels_int = n_channels.item()
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class InvertibleConv1d(torch.nn.Module):
    """
    Invertible 1x1 1-dimensional convolution layer, as used in SqueezeWave
    and WaveGlow. Use reverse_forward to invert weights for inference.
    """

    def __init__(self, n_channels: int):
        super(InvertibleConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.Tensor(n_channels, n_channels).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(n_channels, n_channels, 1)
        self.conv.weight.data = W

    def forward(self, z: torch.Tensor):
        """
        Performs standard forward pass (use for training)
        :param z: of shape (batch_size, group_size, n_groups)
        :return: tuple (Tensor(batch_size, group_size, n_groups), log(det(W)))
        """
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()

        # forward computation
        log_det_W = batch_size * n_of_groups * torch.logdet(W)
        z = self.conv(z)
        return z, log_det_W

    def reverse_forward(self, z: torch.Tensor):
        """
        Performs forward pass on convolution with inverted weights
        (use for inference)
        :param z: of shape (batch_size, group_size, n_groups)
        :return: of shape (batch_size, group_size, n_groups)
        """
        W = self.conv.weight.squeeze()

        # reverse forward computation, cache W_inverse for improved speed
        if not hasattr(self, 'W_inverse'):
            # Reverse computation
            W_inverse = W.inverse()
            if z.dtype == torch.half:
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse[..., None]
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        return z


class DepthwiseSeparableConv1d(torch.nn.Module):
    """
    Clean implementation of depth-wise separable 1d convolution layer proposed
    by SqueezeWave paper as efficient alternative to normal Conv1d in WaveGlow.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(DepthwiseSeparableConv1d, self).__init__()

        assert kernel_size % 2 == 1
        assert in_channels % 2 == 0

        bn = torch.nn.BatchNorm1d(in_channels)
        depthwise = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=in_channels
        )
        pointwise = torch.nn.Conv1d(
            in_channels, out_channels, 1
        )
        self.layer = torch.nn.Sequential(bn, depthwise, pointwise)

    def forward(self, x):
        """
        :param x: of shape (batch_size, in_channels, audio_length)
        :return: of shape (batch_size, out_channels, audio_length)
        """
        x = self.layer(x)
        return x

    @staticmethod
    def remove_batch_norm(module_: "DepthwiseSeparableConv1d") -> "DepthwiseSeparableConv1d":
        depthwise = _fuse_conv_and_bn(module_.layer[1], module_.layer[0])
        pointwise = module_.layer[2]
        module_.layer = torch.nn.Sequential(depthwise, pointwise)
        return module_


class WN(torch.nn.Module):
    """
    Wavenet-like layer for affine coupling to be used by SqueezeWave.

    Notable changes wrt SqueezeWave:
        - adapted for CPU usage (also: simplified & well-documented)
        - extracted hyperparameters: mel_upsample_scale, kernel_size

    Notable changes wrt WaveGlow:
        - mel spectrogram is upsampled independently on every layer
          (as opposed to upsampling before WN in WaveGlow-like models)
        - mel spectrogram is upsampled by interpolation (not ConvTranspose1d)
        - dilation size is constant (1, as opposed to doubling each layer)
        - depthwise separable convolutions are used instead of normal ones

    Notable changes wrt WaveNet:
        - convolutions need not be causal
    """

    def __init__(
            self,
            in_audio_channels: int,
            in_mel_channels: int,
            n_layers: int,
            n_channels: int,
            conv_kernel_size: int,
            mel_upsample_scale: int,
    ):
        super(WN, self).__init__()

        assert conv_kernel_size % 2 == 1
        assert n_channels % 2 == 0

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_layer_channels = 2 * n_channels

        # mel processing layers
        self.cond_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(in_mel_channels, self.n_layer_channels * n_layers, 1),
            name='weight'
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=mel_upsample_scale,
            mode='nearest'
        )

        # audio processing layers
        self.start_conv = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(in_audio_channels, n_channels, 1),
            name='weight'
        )
        self.end_conv = torch.nn.Conv1d(n_channels, 2 * in_audio_channels, 1)
        self.end_conv.weight.data.zero_()
        self.end_conv.bias.data.zero_()

        def _get_depthwise_separable_convolution_layer():
            layer = DepthwiseSeparableConv1d(
                n_channels, self.n_layer_channels, conv_kernel_size
            )
            return layer

        def _get_residual_skip_layer():
            layer = torch.nn.utils.weight_norm(
                torch.nn.Conv1d(n_channels, n_channels, 1), name='weight'
            )
            return layer

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            conv_layer = _get_depthwise_separable_convolution_layer()
            self.in_layers.append(conv_layer)
            res_skip_layer = _get_residual_skip_layer()
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        """
        Forward input:
            - audio_tensor of shape (batch_size, n_in_channels, audio_length)
            - mel_tensor of shape (batch_size, n_mel_channels, mel_length)
        :param forward_input: tuple (audio_tensor, mel_tensor)
        :return: of shape (batch_size, 2*n_in_channels, audio_length)
        """
        audio, mel_spectrogram = forward_input

        msg = "audio and mel_spectrogram should have the same batch size"
        assert audio.shape[0] == mel_spectrogram.shape[0], msg

        audio = self.start_conv(audio)
        mel_spectrogram = self.cond_layer(mel_spectrogram)

        n_channels_tensor = torch.IntTensor([self.n_channels])
        for i in range(self.n_layers):
            layer_offset = i * self.n_layer_channels
            layer_spectrogram = mel_spectrogram[:, layer_offset:layer_offset + self.n_layer_channels, :]

            if audio.size(2) > layer_spectrogram.size(2):
                layer_spectrogram = self.upsample(layer_spectrogram)

            layer_audio = self.in_layers[i](audio)
            acts = fused_add_tanh_sigmoid_multiply(
                layer_audio,
                layer_spectrogram,
                n_channels_tensor
            )
            acts = self.res_skip_layers[i](acts)
            audio = audio + acts
        audio = self.end_conv(audio)
        return audio

    @staticmethod
    def remove_norms(module_: "WN") -> "WN":
        """ Removes batch norm or weight norm from all applicable submodules """
        module_.start = torch.nn.utils.remove_weight_norm(module_.start_conv)
        module_.cond_layer = torch.nn.utils.remove_weight_norm(module_.cond_layer)
        for i, layer_ in enumerate(module_.in_layers):
            layer_ = DepthwiseSeparableConv1d.remove_batch_norm(layer_)
            module_.in_layers[i] = layer_
        for i, layer_ in enumerate(module_.res_skip_layers):
            layer_ = torch.nn.utils.remove_weight_norm(layer_)
            module_.res_skip_layers[i] = layer_
        return module_


class SqueezeWave(torch.nn.Module):
    """
    Our SqueezeWave implementation (refactored from the original one).

    Notable changes wrt SqueezeWave:
        - adapted for CPU usage (also: simplified & well-documented)
        - validation of wn_config parameters
    """

    def __init__(
            self,
            n_flows: int,
            n_audio_channels: int,
            n_mel_channels: int,
            early_return_interval: int,  # n_early_every
            early_return_size: int,  # n_early_size
            wn_config: WNConfig
    ):
        wn_config = asdict(wn_config)
        super(SqueezeWave, self).__init__()

        assert n_audio_channels % 2 == 0
        msg = "WN.in_audio_channels are variable and should not be specified"
        assert "in_audio_channels" not in wn_config, msg
        msg = "WN.in_mel_channels need to be specified explicitly"
        assert "in_mel_channels" not in wn_config, msg
        msg = "Early_return_size must be divisible by 2"
        assert early_return_size % 2 == 0, msg

        self.n_flows = n_flows
        self.n_audio_channels = n_audio_channels
        self.early_return_size = early_return_size
        self.return_early = lambda flow: flow % early_return_interval == 0 and flow > 0

        self.wn_layers = torch.nn.ModuleList()
        self.inv_conv_layers = torch.nn.ModuleList()

        # In WaveGlow, every n_early_every flows we return portion of audio,
        # which decreases the number of channels remaining at these flows
        n_half = n_audio_channels // 2
        n_remaining_channels = n_audio_channels
        for k in range(n_flows):
            if self.return_early(k):
                n_half = n_half - self.early_return_size // 2
                n_remaining_channels = n_remaining_channels - self.early_return_size
            self.inv_conv_layers.append(InvertibleConv1d(n_remaining_channels))
            self.wn_layers.append(WN(n_half, n_mel_channels, **wn_config))

        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input):
        """
        Forward input:
            - mel_spectrogram of shape (batch_size, n_mel_channels, mel_length)
            - audio of shape (batch_size, audio_length)
        :param forward_input: tuple (mel_spectrogram, audio)
        :return: tuple (audio, log_s_list, log_det_W_list)
        """
        mel_spectrogram, audio = forward_input

        audio = audio.unfold(
            1, self.n_audio_channels, self.n_audio_channels
        ).permute(0, 2, 1)

        output_audio_list = []
        log_s_list = []
        log_det_W_list = []
        for k in range(self.n_flows):
            if self.return_early(k):
                output_audio_list.append(audio[:, :self.early_return_size, :])
                audio = audio[:, self.early_return_size:, :]

            audio, log_det_W = self.inv_conv_layers[k](audio)
            log_det_W_list.append(log_det_W)

            half = audio.shape[1] // 2
            audio_0, audio_1 = torch.split(audio, half, dim=1)

            # wn_output ~ audio conditioned on mel_spectrogram
            wn_output = self.wn_layers[k]((audio_0, mel_spectrogram))
            log_s, b = torch.split(wn_output, half, dim=1)

            audio_1 = torch.exp(log_s) * audio_1 + b  # affine coupling
            audio = torch.cat([audio_0, audio_1], 1)
            log_s_list.append(log_s)
        output_audio_list.append(audio)

        output_audio = torch.cat(output_audio_list, 1)
        return output_audio, log_s_list, log_det_W_list

    def infer(self, mel_spectrogram: torch.Tensor, sigma: float = 0.6):
        """
        Use to generate audio from mel spectrogram.
        :param mel_spectrogram: of shape (batch_size, n_mel_channels, mel_length)
        :param sigma: scaling factor for inference, 0.6 used in both WaveGlow and SqueezeWave
        :return:
        """
        l = mel_spectrogram.size(2) * (256 // self.n_audio_channels)
        audio = torch.Tensor(
            mel_spectrogram.size(0),
            self.n_remaining_channels,
            l
        ).normal_()
        if mel_spectrogram.dtype == torch.half:
            audio = audio.half()

        for k in reversed(range(self.n_flows)):
            half = audio.shape[1] // 2
            audio_0, audio_1 = torch.split(audio, half, dim=1)

            # wn_output ~ audio conditioned on mel_spectrogram
            wn_output = self.wn_layers[k]((audio_0, mel_spectrogram))
            s, b = torch.split(wn_output, half, dim=1)

            # reverse affine coupling, convolution and early return
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)
            audio = self.inv_conv_layers[k].reverse_forward(audio)
            if self.return_early(k):
                z = torch.Tensor(
                    mel_spectrogram.size(0), self.early_return_size, l
                ).normal_()
                if mel_spectrogram.dtype == torch.half:
                    z = z.half()
                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
        return audio

    @staticmethod
    def remove_norms(model: "SqueezeWave") -> "SqueezeWave":
        """ Removes batch norm or weight norm from all applicable submodules """
        squeeze_wave = model
        for i, wn_layer in enumerate(squeeze_wave.wn_layers):
            squeeze_wave.wn_layers[i] = WN.remove_norms(wn_layer)
        return squeeze_wave


def _fuse_conv_and_bn(
        conv: torch.nn.Conv1d,
        bn: torch.nn.BatchNorm1d
) -> torch.nn.Conv1d:
    fused_conv = torch.nn.Conv1d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        padding=conv.padding,
        bias=True,
        groups=conv.groups
    )

    # fuse weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    w_bn = w_bn.clone()
    fused_conv.weight.data = torch.mm(w_bn, w_conv).view(fused_conv.weight.size())

    # fuse bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    b_bn = torch.unsqueeze(b_bn, 1)
    bn_3 = b_bn.expand(-1, 3)
    b = torch.matmul(
        w_conv, torch.transpose(bn_3, 0, 1)
    )[range(b_bn.size()[0]), range(b_bn.size()[0])]
    fused_conv.bias.data = (b_conv + b)

    return fused_conv


# TODO: How do we want to organize tests?
if __name__ == '__main__':
    wn_config = WNConfig(8, 256, 3, 2)
    sw = SqueezeWave(12, 128, 80, 2, 16, wn_config)
    sw = SqueezeWave.remove_norms(sw)
    sw = sw.eval()

    n_benchmarks = 3
    total_time = 0
    for _ in range(n_benchmarks):
        zero_mel = torch.zeros(24, 80, 1024)
        start = time()
        with torch.no_grad():
            audio = sw.infer(zero_mel)
        end = time()
        total_time += end - start
        print(f"{total_time = :.2f}")
    avg_time = total_time / n_benchmarks

    expected_gen_rate = 123000 / 22050  # based on largest model in the paper
    audio_len_in_s = 1024 * 256 / 22050
    expected_avg_time = audio_len_in_s / expected_gen_rate

    slowdown_rate = avg_time / expected_avg_time
    print(f"{slowdown_rate = :.4f}: {avg_time = :.2f}, {expected_avg_time = :.2f}")
