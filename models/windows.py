import torch
from utils.nest_model import NestedTensor
import math
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np

'''
尺度窗口特征提取（Windows Wrapper / Multi-scale Windowing）
在视觉任务中，要做高分辨率目标检测（比如 4K 图像里寻找极其微小的螺丝钉），
直接把 4K 图像喂进 Transformer 会导致显存瞬间爆炸（Attention 复杂度是 O(N^2)）。
如果你把 4K 强行缩小（Resize）喂进去，那些微小的螺丝钉又会丢失所有的像素细节。

WindowsWrapper 就是为了解决高分辨率输入与显存/细节保留之间的矛盾而诞生的。

'''
__all__ = [
    'WindowsWrapper'
]
class WindowsWrapper(torch.nn.Module):
    """
    This wrapper will take an input (NestedTensor) at size (h, w) and split it
    in `N = n_windows_h * n_windows_w` equally sized windows (the bottom and right windows might
    be a little bit smaller), with sizes that are multiples of the patch size (as the input should be).

    Then, the input will be resized at the size of the top left window (h / n_windows_h, w / n_windows_w).
    This resized input, plus the N windows, will be passed through the backbone.
    Then, the features of the resized input will be resized to the original input size, while the
    features of the windows will be concatenated side by side to reconstruct a feature map also
    corresponding to the original image's size.

    Finally, both the features from the windows and from the resized images are stacked.
    Compared to the output of the backbone of size [B, C, H, W], the output here is [B, 2 * C, H, W]
    """

    def __init__(self, backbone, n_windows_w, n_windows_h, patch_size):
        # Assuming image size is divisible by patch_size
        super().__init__()
        self._backbone = backbone
        self._n_windows_w = n_windows_w # 宽度的窗口数
        self._n_windows_h = n_windows_h # 高度的窗口数
        self._patch_size = patch_size
        self.strides = backbone.strides
        self.num_channels = [el * 2 for el in backbone.num_channels]  # resized + windows

    def forward(self, tensor_list: NestedTensor):
        tensors = tensor_list.tensors
        original_h, original_w = tensors.shape[2], tensors.shape[3]
        # Get height and width of the windows, such that it is a multiple of the patch size
        # 将图片切成几个窗口，每个窗口的大小的长或者宽是patch_size的整数倍，即这里的window_h，window_w必须是patch_size的整数倍
        window_h = math.ceil((original_h // self._n_windows_h) / self._patch_size) * self._patch_size
        window_w = math.ceil((original_w // self._n_windows_w) / self._patch_size) * self._patch_size
        all_h = [window_h] * (self._n_windows_h - 1) + [original_h - window_h * (self._n_windows_h - 1)] # 所有的窗口高度，为了保证总高度等于原图高度，最后一个窗口（最下边）通常会比前面的窗口小一点
        all_w = [window_w] * (self._n_windows_w - 1) + [original_w - window_w * (self._n_windows_w - 1)] # 所有的窗口宽度，为了保证总宽度等于原图宽度，最后一个窗口（最右边）通常会比前面的窗口小一点
        all_h_cumsum = [0] + list(np.cumsum(all_h))
        all_w_cumsum = [0] + list(np.cumsum(all_w))
        window_patch_features = [[0 for _ in range(self._n_windows_w)] for _ in range(self._n_windows_h)]
        # 用 v2.functional.crop 把大图硬生生切成了一块块的小图
        # 把每一块小图独立地送进了 self._backbone（也就是 DINOv3）去提取特征
        for ih in range(self._n_windows_h):
            for iw in range(self._n_windows_w):
                window_tensor = v2.functional.crop(
                    tensors, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_mask = v2.functional.crop(
                    tensor_list.mask, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_patch_features[ih][iw] = self._backbone(NestedTensor(tensors=window_tensor, mask=window_mask))[0]
        # 等所有小块都提取完特征后，再按照原本的位置（上、下、左、右）把这些特征图拼装回一个完整的大特征图（window_tensors）
        window_tensors = torch.cat(
            [
                torch.cat([el.tensors for el in window_patch_features[ih]], dim=-1)  # type: ignore
                for ih in range(len(window_patch_features))
            ],
            dim=-2,
        )
        # Also compute the global features in a "preferential" setting, of lower resolution
        # 又把整张原图做了一次强行 resize，缩小到了一个单独窗口的大小
        resized_global_tensor = v2.functional.resize(tensors, size=(window_h, window_w))
        # 将缩放的原图送入backbone，得到global特征
        global_features = self._backbone(
            NestedTensor(tensors=resized_global_tensor, mask=tensor_list.mask)
        )  # mask is not used
        # 将global特征缩放成原来的分辨率，再将其与高分辨率的window特征做拼接
        concat_tensors = torch.cat(
            [v2.functional.resize(global_features[0].tensors, size=window_tensors.shape[-2:]), window_tensors], dim=1
        )
        global_mask = F.interpolate(tensor_list.mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        out = [NestedTensor(tensors=concat_tensors, mask=global_mask)]
        return out