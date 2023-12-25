# coding=utf-8
import torch
from torch import nn

from timm.models.layers import trunc_normal_

from detrex.layers import FFN, TransformerLayerSequence, BaseTransformerLayer, MultiheadAttention
from .flow_attention import LinearAttention, LinearCrossAttention


class FlowDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            attn_dropout: float = 0.1,
            feedforward_dim: int = 2048,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            post_norm: bool = True,
            batch_first: bool = False,
            proj_drop: float = 0.0,
            competition_temperature: float = 1,
            iter_fact: float = 1
    ):
        super(FlowDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=LinearAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                    competition_temperature=competition_temperature,
                    iter_fact=iter_fact,
                    proj_drop=proj_drop
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
            self,
            query,
            key,
            value,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class FlowDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            attn_dropout: float = 0.1,
            feedforward_dim: int = 2048,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            post_norm: bool = True,
            return_intermediate: bool = True,
            batch_first: bool = False,
            fine_turning: bool = False,
            large_scale: bool = False,
            proj_drop: float = 0.0,
            competition_temperature: float = 1,
            iter_fact: int = 1
    ):
        super(FlowDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ) if not large_scale else [
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                        proj_drop=proj_drop
                    ),
                    LinearCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                        iter_fact=iter_fact,
                        proj_drop=proj_drop,
                    )
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )

        # fine turing
        # for layer in self.layers:
        #     # freeze decoder self attention for fine turing
        #     layer.attentions[0].requires_grad_(fine_turning)
        #     # freeze the layer norm between decoder self attention and decoder cross attention
        #     layer.norms[0].requires_grad_(fine_turning)
        # freeze decoder self attention for fine turing
        # self.layers[0].attentions[0].requires_grad_(not fine_turning)
        # freeze the layer norm between decoder self attention and decoder cross attention
        # self.layers[0].norms[0].requires_grad_(not fine_turning)

        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
            self,
            query,
            key,
            value,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):

        if not self.return_intermediate:
            for layer in self.layers:
                query = layer(
                    query,  # target
                    key,  # memory
                    value,  # memory
                    query_pos=query_pos,  # pos_embed
                    key_pos=key_pos,  # query_embed
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,  # mask
                    **kwargs,
                )

            if self.post_norm_layer is not None:
                query = self.post_norm_layer(query)[None]
            return query

        # return intermediate
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,  # target
                key,  # memory
                value,  # memory
                query_pos=query_pos,  # pos_embed
                key_pos=key_pos,  # query_embed -> query_embed.weights
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,  # mask
                **kwargs,
            )

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        return torch.stack(intermediate)


class FlowDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(FlowDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.contiguous().view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.contiguous().view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1
        )  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.contiguous().view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )
        decoder_output = decoder_output.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return decoder_output, memory
