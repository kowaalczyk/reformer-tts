import torch


class SqueezeWaveLoss(torch.nn.Module):
    """ Loss for SqueezeWave model """

    def __init__(self, sigma=1.0):
        """
        :param sigma: scaling factor for training, 1.0 used in both WaveGlow and SqueezeWave
        """
        super(SqueezeWaveLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        """
        :param model_output: tuple(audio, log_s_list, log_det_W_list)
        :return: loss: float
        """
        z, log_s_list, log_det_W_list = model_output

        log_s_sum = 0.
        log_det_W_sum = 0.
        for i, log_s in enumerate(log_s_list):
            log_s_sum += torch.sum(log_s).double()
            log_det_W_sum += log_det_W_list[i].double()

        z = z.double()
        loss = torch.sum(z * z) / (2 * self.sigma ** 2) - log_s_sum - log_det_W_sum
        loss = loss / (z.size(0) * z.size(1) * z.size(2))

        return loss.float()
