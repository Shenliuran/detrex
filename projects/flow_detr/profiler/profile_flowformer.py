import torch
import torch.nn as nn

from detrex.layers import FFN, BaseTransformerLayer, TransformerLayerSequence, MultiheadAttention, MLP
from projects.flow_detr.modeling.flow_attention import LinearAttention, LinearCrossAttention

sequence = TransformerLayerSequence(
        transformer_layers=BaseTransformerLayer(
            attn=[
                LinearCrossAttention(256, 8),
                MultiheadAttention(256, 8),
            ],
            ffn=FFN(256, 1024, num_fcs=2, activation=nn.GELU()),
            norm=nn.LayerNorm(256),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
        num_layers=8
    )