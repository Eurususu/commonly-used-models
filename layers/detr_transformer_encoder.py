import torch.nn as nn
from utils.misc import _get_clones, _get_activation_fn
from typing import Optional
from torch import Tensor


__all__ = ['TransformerEncoder', 'TransformerEncoderLayer']

'''
encoder_layer 是什么？ 它是一层完整的单层 Transformer 结构（包含了一个多头自注意力机制 Multi-Head Attention 和一个前馈神经网络 FFN）
_get_clones: _get_clones 是一个深拷贝（Deepcopy）工具，它把这一层完美的复制了 num_layers 份（比如 6 份）。这 6 层结构一模一样，但它们各自拥有独立可学习的参数矩阵。
norm： 大楼的封顶工作。通常是一个 LayerNorm，用于稳定最后一层的输出分布。
'''
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

'''
TransformerEncoderLayer就是上面TransformerEncoder中的encoder_layer

'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        # Keeping Dropout 0 in self attention as it makes the eval 10% faster without performance change
        # 把这个 Dropout 设为 0，对最终精度没有任何负面影响，反而能让推理（Eval）速度白白提升 10%！ 因为底层 CUDA 算子少了一层毫无意义的随机数生成和内存拷贝。这是纯粹的工业界实战经验。
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    '''
    Post-Norm
    运算（Attention/FFN） ➡️ 加上残差（+ src） ➡️ 最后归一化（LayerNorm）。
    这种结构最终的精度上限往往非常高，但是极难训练！因为它在深层网络中容易发生梯度消失或爆炸，通常需要配合极其复杂的学习率预热（Learning Rate Warmup）策略才能收敛。
    '''
    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # 让坐标只参与寻址计算，不参与特征表达。 所以只有q和k加上了pos，而v没有加pos
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    '''
    Pre-Norm
    先归一化（LayerNorm） ➡️ 运算（Attention/FFN） ➡️ 加上残差（+ src）
    它把 LayerNorm 放在了进入 Attention 和 FFN 之前：

    在 Pre-Norm 中，主干道（残差连接 src + ...）上没有任何阻碍！不管网络叠了 100 层还是 1000 层，
    最底层的梯度都可以不受 LayerNorm 的干扰，通过加法直接畅通无阻地传回第一层。这种设计极大地增强了训练的稳定性，几乎不需要 Learning Rate Warmup 就能快速收敛。
    '''
    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    '''
    任务是对接 DETR 的检测头，通常采用 Pre-Norm (normalize_before=True) 会让你的网络在训练早期表现得更加平滑和听话。
    '''
    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)