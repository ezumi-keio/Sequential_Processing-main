import torch
import numpy as np
from scipy import stats
import torch.nn as nn
import cv2

# Criteria

crit_lst = ['PSNR', 'SSIM', 'LPIPS']


def return_crit_func(name, opts):
    assert (name in crit_lst), 'NOT SUPPORTED YET!'
    crit_func_cls = globals()[name]
    if opts is not None:
        return crit_func_cls(**opts)
    else:
        return crit_func_cls()


class PSNR(nn.Module):
    """Input tensor. Return a float."""

    def __init__(self):
        super().__init__()

        self.mse_func = nn.MSELoss()
        self.lsb = False  # lower is better

    def forward(self, x, y):
        mse = self.mse_func(x, y)
        psnr = 10 * torch.log10(1. / mse)
        return psnr.item()


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity.

    Args:
        if_spatial: return a score or a map of scores.

    https://github.com/richzhang/PerceptualSimilarity
    """

    def __init__(self, net='alex', if_spatial=False, if_cuda=True):
        super().__init__()
        import lpips

        self.lpips_fn = lpips.LPIPS(net=net, spatial=if_spatial)
        if if_cuda:
            self.lpips_fn.cuda()

        self.lsb = True

    @staticmethod
    def _preprocess(inp, mode):
        out = None
        if mode == 'im':
            im = inp[:, :, ::-1]  # (H W BGR) -> (H W RGB)
            im = im / (255. / 2.) - 1.
            im = im[..., np.newaxis]  # (H W RGB 1)
            im = im.transpose(3, 2, 0, 1)  # (B=1 C=RGB H W)
            out = torch.Tensor(im)
        elif mode == 'tensor':
            out = inp * 2. - 1.
        return out

    def forward(self, ref, im):
        """
        im: cv2 loaded images, or ([RGB] H W), [0, 1] CUDA tensor.
        """
        mode = 'im' if ref.dtype == np.uint8 else 'tensor'
        ref = self._preprocess(ref, mode=mode)
        im = self._preprocess(im, mode=mode)
        lpips_score = self.lpips_fn.forward(ref, im)
        return lpips_score.item()


# Others

class PCC:
    """Pearson correlation coefficient."""

    def __init__(self):
        self.help = (
            'Pearson correlation coefficient measures linear correlation '
            'between two variables X and Y. '
            'It has a value between +-1. '
            '+1: total positive linear correlation. '
            '0: no linear correlation. '
            '-1: total negative linear correlation. '
            'See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient'
        )

    @staticmethod
    def cal_pcc_two_imgs(x, y):
        """Calculate Pearson correlation coefficient of two images.

        Consider each pixel in x as a sample from a variable X, each pixel in y
        as a sample from a variable Y. Then an mxn image equals to mxn times
        sampling.

        Input:
            x, y: two imgs (numpy array).
        Return:
            (cc value, p-value)

        Formula: https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.stats.pearsonr.html?highlight=pearson#scipy.stats.pearsonr

        Note: x/y should not be a constant! Else, the sigma will be zero,
        and the cc value will be not defined (nan).
        """
        return stats.pearsonr(x.reshape((-1,)), y.reshape((-1,)))

    def _test(self):
        x = np.array([[3, 4], [1, 1]], dtype=np.float32)
        y = x + np.ones((2, 2), dtype=np.float32)
        print(self.cal_pcc_two_imgs(x, y))


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _ssim(img1, img2):
        """Calculate SSIM (structural similarity) for one channel images.

        It is called by func:`calculate_ssim`.

        Args:
            img1 (ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (ndarray): Images with range [0, 255] with order 'HWC'.

        Returns:
            float: ssim result.
        """

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, img1_in, img2_in):
        """Calculate SSIM (structural similarity).

        Ref:
        Image quality assessment: From error visibility to structural similarity

        The results are the same as that of the official released MATLAB code in
        https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        For three-channel images, SSIM is calculated for each channel and then
        averaged.

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These
                pixels are not involved in the SSIM calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: ssim result.
        """
        device1, device2 = img1_in.device, img2_in.device
        img1 = (torch.permute(img1_in, (1,2,0)).to('cpu').detach().numpy().copy() * 255).astype(np.uint8)
        img2 = (torch.permute(img2_in, (1,2,0)).to('cpu').detach().numpy().copy() * 255).astype(np.uint8)
        assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        ssims = []
        for i in range(img1.shape[2]):
            res = self._ssim(img1[..., i], img2[..., i])
            ssims.append(res)
        return np.array(ssims).mean()

