from math import sqrt
import torch.nn as nn
import torch
from models.my_method.Attentions.attention_masking import TriangularCausalMask
import numpy as np
from einops import einsum


class DozerAttention(nn.Module):
    def __init__(self, local_window, stride, rand_rate, vary_len, pred_len, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(DozerAttention, self).__init__()
        self.scale = scale

        self.local_window = local_window
        self.stride = stride
        self.rand_rate = rand_rate
        self.vary_len = vary_len
        self.mask_flag = mask_flag
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, queries, keys, values, attn_mask):
        # Batch size, Seq len, Head, dim/head
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        scale = self.scale or 1. / sqrt(D)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        sparse_mask = torch.zeros(L_Q, L_K, device=scores.device)
        # Self Attention
        if L_Q == L_K:
            if self.local_window:
                for w_idx in range(self.local_window//2+1):
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)

            if self.stride:
                stride = self.stride + 1
                for w_idx in range(0, L_Q, stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)

        # Cross Attention
        if L_Q != L_K:
            # 1. local
            if self.local_window:
                local_window = self.local_window//2 if self.local_window>1 else self.local_window
                sparse_mask[:, -local_window:] = 1

            # 2. Stride
            if self.stride:
                start_index = L_K - L_Q//2
                stride = self.stride + 1
                for w_idx in range(start_index, L_K, stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask,
                                                         torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
                                                         w_idx)
                for w_idx in range(start_index, -L_K, -stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask,
                                                         torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
                                                         w_idx)

            if self.vary_len or type(self.vary_len) is int:
                start_index = -self.pred_len+self.vary_len-1
                var_len_mask = torch.tril(torch.ones(L_Q, L_K, device=scores.device), diagonal=start_index)
                var_len_mask = torch.flip(var_len_mask, [1])
                sparse_mask = torch.where((sparse_mask + var_len_mask) >= 1, 1, 0)

        scores = scores * sparse_mask

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
            # attn_mask is bool
            scores.masked_fill_(attn_mask.mask, -np.inf)
        b = scores[0, 0, :, :].detach().cpu().numpy()
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



class DozerAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(DozerAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        x = torch.clone(queries)
        # Batch size, Seq len, embed_dim
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Batch size, Seq len, head, embed_dim/head
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            x,
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn