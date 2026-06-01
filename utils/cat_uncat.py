import torch
from torch import Tensor
from typing import List, Tuple

'''
全连接层（Linear / MLP）通常只关心最后一个维度（特征维度 C），前面的维度不管是 (Batch, Length) 还是 (Batch, Height, Width)
对线性层来说都是独立的“Token”。

cat_keep_shapes（打包）
这个函数的任务是把 x_list（比如包含了不同分辨率特征图的列表）压扁并拼在一起。
shapes: 记录下每个 Tensor 原本的样子，为了以后还原。
num_tokens: 这里用了一个非常巧妙的写法 x.select(dim=-1, index=0).numel()。意思是抛弃最后一个特征维度（C），
计算前面所有维度乘起来的总长度（即 Token 的总数量）。例如 (2, 14, 14, 768) 会算出 2 * 14 * 14 = 392。
flattened: 将每个 Tensor 展平成 (num_tokens, C) 的形状，并在第 0 维度上拼接。
多个形状各异的 Tensor 变成了一个巨大的 (Total_Tokens, C) 的二维矩阵。

uncat_with_shapes（解包）
等网络计算完之后，把那个巨大的二维矩阵拆回去。
outputs_splitted: torch.split_with_sizes根据打包时记录的 num_tokens，把大矩阵重新切分成多个小矩阵。
shapes_adjusted:  网络前向传播可能会改变特征维度（比如把 C 映射成了 C_{out}）。
所以还原时，不能完全照搬原始 shape，而是保留前面的空间维度 shape[:-1]，把最后一个维度替换成网络输出的真实维度 flattened.shape[-1]
reshape: 最后把每个小块 reshape 回原始的多维结构
'''
def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped