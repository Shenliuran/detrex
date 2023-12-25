import pytest
import torch
import torchvision

from projects.flow_detr.modeling.flow_attention import LinearAttention, LinearCrossAttention


@pytest.mark.parametrize("bs", [16])
@pytest.mark.parametrize("l", [3400])
@pytest.mark.parametrize("c", [256])
@pytest.mark.parametrize("batch_first", [False])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("competition_temperature", [0.05, 0.5, 1, 1.5, 2])
def test_flow_self_attention(nheads, l, bs, c, batch_first, competition_temperature):
    d_model = 256

    if batch_first:
        input = torch.randn(bs, l, c)
        query_pos = torch.randn(bs, l, c)
    else:
        input = torch.randn(l, bs, c)
        query_pos = torch.randn(l, bs, c)

    key_padding_mask = torch.zeros(bs, l, dtype=torch.bool)  # (bs, n)
    flow_attention_detrex = LinearAttention(embed_dim=d_model, num_heads=nheads, batch_first=batch_first,
                                            competition_temperature=competition_temperature)

    detrex_output = flow_attention_detrex(query=input, key=input, value=input, query_pos=query_pos, key_pos=query_pos,
                                          key_padding_mask=key_padding_mask)

    if batch_first:
        assert detrex_output.shape == torch.Size([bs, l, c])
    else:
        assert detrex_output.shape == torch.Size([l, bs, c])


@pytest.mark.parametrize("bs", [16])
@pytest.mark.parametrize("num_query", [100])
@pytest.mark.parametrize("memory_l", [3400])
@pytest.mark.parametrize("c", [256])
@pytest.mark.parametrize("batch_first", [False])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("d_model", [256])
@pytest.mark.parametrize("competition_temperature", [0.05, 0.5, 1, 1.5, 2])
def test_flow_cross_attention(num_query, memory_l, nheads, d_model, bs, c, batch_first,
                              competition_temperature):
    memory = torch.randn(memory_l, bs, c)
    pos_embed = torch.randn(memory_l, bs, c)
    query_embed = torch.randn(num_query, bs, c)
    if batch_first:
        memory = memory.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.permute(1, 0, 2)
    target = torch.ones_like(query_embed)

    mask = torch.zeros(bs, memory_l, dtype=torch.bool)  # (bs, n)
    flow_attention_detrex = LinearCrossAttention(embed_dim=d_model, num_heads=nheads, batch_first=batch_first,
                                            competition_temperature=competition_temperature)

    detrex_output = flow_attention_detrex(query=target, key=memory, value=memory, query_pos=query_embed,
                                          key_pos=pos_embed,
                                          key_padding_mask=mask)

    if batch_first:
        assert detrex_output.shape == torch.Size([bs, num_query, c])
    else:
        assert detrex_output.shape == torch.Size([num_query, bs, c])
