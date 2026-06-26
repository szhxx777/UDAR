import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class UDARAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False, period=25, dynamic_temp=True, use_period=True):
        super(UDARAttention, self).__init__()
        self.dynamic_temp = dynamic_temp
        self.use_period = use_period
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        range_vec = torch.arange(win_size).float()
        dist_mat = torch.abs(range_vec.unsqueeze(1) - range_vec.unsqueeze(0))

        if use_period:
            period_vec = range_vec % period
            p_dist_mat = torch.abs(period_vec.unsqueeze(1) - period_vec.unsqueeze(0))
            blended = dist_mat + p_dist_mat
        else:
            blended = dist_mat

        self.register_buffer('blended_distances', blended)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)

        D_blended_expanded = self.blended_distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-D_blended_expanded ** 2 / 2 / (sigma ** 2))

        uncertainty = torch.mean(sigma, dim=-1, keepdim=True)
        uncertainty_normalized = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

        if self.dynamic_temp:
            temperature = 0.5 + 1.5 * uncertainty_normalized
        else:
            temperature = 1.0

        series = self.dropout(torch.softmax(attn / temperature, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, use_fusion=True):
        super(AttentionLayer, self).__init__()
        self.use_fusion = use_fusion

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.sigma_projection_fine = nn.Linear(d_model, n_heads)
        self.sigma_projection_coarse = nn.Linear(d_model, n_heads)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * n_heads, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_heads),
            nn.Sigmoid()
        )

    def forward(self, queries, keys, values, attn_mask, x_fine, x_coarse):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        sigma_fine = self.sigma_projection_fine(x_fine).view(B, L, H)

        if self.use_fusion:
            sigma_coarse = self.sigma_projection_coarse(x_coarse).view(B, L, H)
            sigma_fused_input = torch.cat([sigma_fine, sigma_coarse], dim=-1).view(-1, 2 * H)
            alpha = self.fusion_mlp(sigma_fused_input).view(B, L, H)
            sigma_fused = alpha * sigma_fine + (1.0 - alpha) * sigma_coarse
        else:
            sigma_fused = sigma_fine

        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma_fused, attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
