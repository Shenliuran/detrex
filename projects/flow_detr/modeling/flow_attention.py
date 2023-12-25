import warnings

import torch
import torch.nn as nn
from typing import Optional, Union
import numpy as np


class LinearAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            competition_temperature: float = 1,
            iter_fact: int = 1,
            eps: float = 1e-6,
            **kwargs,
    ):
        super(LinearAttention, self).__init__()

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        # self.learnable_temperature = nn.Parameter(torch.ones(2, 2))
        self.learnable_temperature = nn.Parameter(torch.FloatTensor([1.]))

        # self.attn_drop = nn.Dropout(attn_drop)
        self.out_projection_drop = nn.Dropout(proj_drop)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        # self.competition_temperature = competition_temperature
        # self.iter_fact = iter_fact
        self.eps = eps

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        # if self.competition_temperature is not None:
        #     if self.iter_fact is None:
        #         T = self.competition_temperature
        #     else:
        #         T = self.competition_temperature / (
        #                 1 + np.log(current_iter * self.iter_fact + 1))
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )

        if self.batch_first:  # (b n c)
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            identity = identity.transpose(0, 1)

        query_content = self.query_projection(query)
        key_content = self.key_projection(key)
        value = self.value_projection(value)

        N, B, C = query.shape
        HW, _, _ = key_content.shape

        q = query_content + query_pos if query_pos is not None else query_content
        k = key_content + key_pos if key_pos is not None else key_content
        v = value

        q = q.view(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)  # (b, h, n, c)
        k = k.view(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.view(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        q = self.kernel_method(q)
        k = self.kernel_method(k)

        if key_padding_mask is not None:
            v.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(-1), float(0))

        # (1) Calculate incoming and outgoing flow
        # 流汇入 I = Q ⊙ Broadcast(Sum(K,dim=0),dim=0) 在每一个head内部做求和，K(dim=0)对应是keys(dim=2)
        sink_incoming = 1.0 / (self.sum(q + self.eps, k.sum(dim=2) + self.eps) + self.eps)
        source_outgoing = 1.0 / (self.sum(k + self.eps, q.sum(dim=2) + self.eps) + self.eps)

        # (2) conservation refine for source and sink
        # 汇入流守恒
        conserved_sink = self.sum(q + self.eps, (k * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps) + self.eps
        # 源流出守恒
        conserved_source = self.sum(k + self.eps, (q * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps) + self.eps
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability

        # (3)
        # Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # Competition
        source_competition = torch.softmax(torch.div(conserved_source, self.learnable_temperature), dim=-1) * float(
            key.shape[2])

        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        out_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]

        out = out_update.reshape(B, N, C)
        out = self.out_projection(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.out_projection_drop(out)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)


class LinearCrossAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            competition_temperature: float = 1,
            iter_fact: int = 1,
            eps: float = 1e-6,
            **kwargs,
    ):
        super(LinearCrossAttention, self).__init__()

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

        # self.attn_drop = nn.Dropout(attn_drop)
        self.out_projection_drop = nn.Dropout(proj_drop)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        # self.competition_temperature = competition_temperature
        # self.iter_fact = iter_fact
        self.eps = eps


    def kernel_method(self, x):
        return torch.sigmoid(x)

    def sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.einsum("nhld,nhld->nhl", a, b)
        # return torch.sum(a[:, :, :, None] * b[:, :, :, None], dim=-1)

    def causal_dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhldm", k, v)
        kv = torch.cumsum(kv, dim=2)
        qkv = torch.einsum("nhld,nhldm->nhlm", q, kv)
        return qkv

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        # if self.competition_temperature is not None:
        #     if self.iter_fact is None:
        #         T = self.competition_temperature
        #     else:
        #         T = self.competition_temperature / (
        #                 1 + np.log(current_iter * self.iter_fact + 1))
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )

        if self.batch_first:  # (b n c)
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            identity = identity.transpose(0, 1)

        N, B, C = query.shape
        HW, _, _ = key.shape

        query_content = self.query_projection(query)
        key_content = self.key_projection(key)
        value = self.value_projection(value)

        q = query_content + query_pos if query_pos is not None else query_content
        k = key_content + key_pos if key_pos is not None else key_content
        v = value

        q = q.view(N, B, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)  # (b, n, h, c)
        k = k.view(HW, B, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        v = v.view(HW, B, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)

        q = self.kernel_method(q)
        k = self.kernel_method(k)

        if HW > N:
            pad_zeros = torch.zeros(HW - N, B, self.num_heads, C // self.num_heads, device=q.device)
            q = torch.cat([q.permute(1, 0, 2, 3), pad_zeros]).permute(1, 0, 2, 3)

        ## 3. Causal Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (self.sum(q + self.eps, k.cumsum(dim=2) + self.eps))
        source_outgoing = 1.0 / (self.sum(k + self.eps, q.cumsum(dim=2) + self.eps))
        # approximate normal conservation col and row by multiplying corresponding element number
        # normal = ((torch.arange(q.shape[2])).float() + 1.0).to(q.device)[None, None, :]
        normal = ((torch.arange(q.shape[2], device=q.device)).float() + 1.0)[None, None, :]
        sink_incoming = sink_incoming * normal
        source_outgoing = source_outgoing * normal
        # (2) conservation refine for source and sink
        conserved_sink = self.sum(q + self.eps,
                                  (k * source_outgoing[:, :, :, None]).cumsum(dim=2) + self.eps) / normal
        conserved_source = self.sum(k + self.eps,
                                    (q * sink_incoming[:, :, :, None]).cumsum(
                                        dim=2) + self.eps) / normal
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink)
        conserved_source = torch.exp(conserved_source)
        source_competition = (conserved_source / conserved_source.cumsum(dim=-1)) * normal
        # (4) Causal dot product
        out_update = (self.causal_dot_product(q * (sink_incoming[:, :, :, None] / normal[:, :, :, None]),
                                              # for value normalization
                                              k,
                                              v * source_competition[:, :, :, None])  # competition
                      * sink_allocation[:, :, :, None])[:, :N, ...]  # allocation
        out = out_update.transpose(1, 2).reshape(B, N, C)
        out = self.out_projection(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.out_projection_drop(out)
