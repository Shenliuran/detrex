import pytest
import torch
import torch.nn as nn

from detrex.layers import FFN, BaseTransformerLayer, TransformerLayerSequence, MultiheadAttention, MLP
from projects.flow_detr.modeling.flow_attention import LinearAttention, LinearCrossAttention


def test_ffn():
    with pytest.raises(AssertionError):
        FFN(num_fcs=1)

    ffn = FFN(ffn_drop=0.0)
    input_tensor = torch.rand(2, 20, 256)
    input_tensor_nbc = input_tensor.transpose(0, 1)
    assert torch.allclose(ffn(input_tensor).sum(), ffn(input_tensor_nbc).sum())
    residual = torch.rand_like(input_tensor)
    torch.allclose(
        ffn(input_tensor, identity=residual).sum(),
        ffn(input_tensor).sum() + residual.sum() - input_tensor.sum(),
    )


@pytest.mark.parametrize("embed_dim", [256])
def test_basetransformerlayer(embed_dim):
    attn = LinearAttention(embed_dim=embed_dim, num_heads=8)
    ffn = FFN(embed_dim, 1024, num_fcs=2, activation=nn.ReLU(inplace=True))
    base_layer = BaseTransformerLayer(
        attn=attn,
        ffn=ffn,
        norm=nn.LayerNorm(embed_dim),
        operation_order=("self_attn", "norm", "ffn", "norm"),
    )
    feedforward_dim = 1024

    assert attn.batch_first is False
    assert base_layer.ffns[0].feedforward_dim == feedforward_dim
    base_layer.ffns.requires_grad_(False)
    base_layer.norms.requires_grad_(False)
    print(base_layer)
    for param in base_layer.parameters():
        print(param.requires_grad)

    # in_tensor = torch.rand(2, 10, embed_dim)
    # query_pos = torch.randn(16, 1, 256)
    # base_layer(in_tensor, query_pos)


def test_flowformer_encoder():
    sequence = TransformerLayerSequence(
        transformer_layers=BaseTransformerLayer(
            attn=LinearAttention(256, 8),
            ffn=FFN(256, 1024, num_fcs=2),
            norm=nn.LayerNorm(256),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        ),
        num_layers=8
    )

    assert sequence.layers[0].pre_norm is False
    # for layer in sequence.layers:
    #     layer.ffns.requires_grad_(False)
    #     layer.norms.requires_grad_(False)
    #     print("\n", layer.ffns[0])

    # print("\n", list(n for n, _ in sequence.layers[0].attentions.named_parameters()))

    # assert sequence.layers[6].pre_norm is False
    # for param in sequence.parameters():
    #     print(param.requires_grad)


def test_flowformer_decoder():
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
    for layer in sequence.layers:
        layer.attentions[1].requires_grad_(False)
        layer.norms[0].requires_grad_(False)

    for param in sequence.parameters():
        print(param.requires_grad)

    assert sequence.num_layers == 8
    assert sequence.layers[0].embed_dim == 256
    assert sequence.layers[0].pre_norm is False
    with pytest.raises(AssertionError):
        TransformerLayerSequence(
            transformer_layers=[
                BaseTransformerLayer(
                    attn=[
                        LinearAttention(256, 8),
                        MultiheadAttention(256, 8),
                    ],
                    ffn=FFN(256, 1024, num_fcs=2),
                    norm=nn.LayerNorm(256),
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                ),
            ],
            num_layers=6,
        )


def test_transformerlayersequence():
    sequence = TransformerLayerSequence(
        transformer_layers=BaseTransformerLayer(
            attn=[
                LinearAttention(256, 8, batch_first=True),
                MultiheadAttention(256, 8, batch_first=True),
            ],
            ffn=FFN(256, 1024, num_fcs=2),
            norm=nn.LayerNorm(256),
            operation_order=("norm", "self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
        num_layers=6,
    )
    assert sequence.num_layers == 6
    assert sequence.layers[0].embed_dim == 256
    assert sequence.layers[0].pre_norm is True
    with pytest.raises(AssertionError):
        TransformerLayerSequence(
            transformer_layers=[
                BaseTransformerLayer(
                    attn=[
                        MultiheadAttention(256, 8, batch_first=True),
                        MultiheadAttention(256, 8, batch_first=True),
                    ],
                    ffn=FFN(256, 1024, num_fcs=2),
                    norm=nn.LayerNorm(256),
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                ),
            ],
            num_layers=6,
        )
