import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AttentionLayer, UDARAttention
from .embed import DataEmbedding, StandardDecomposition, UncertaintyDecomposition


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_coarse, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            x_fine=x,
            x_coarse=x_coarse
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, x_coarse, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, x_coarse, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class UDAR(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, decomp_kernel=25,
                 period=25):
        super(UDAR, self).__init__()

        self.output_attention = output_attention
        self.decomp = UncertaintyDecomposition(decomp_kernel, period)

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        UDARAttention(
                            win_size,
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                            period=period,
                            dynamic_temp=True,
                            use_period=True,
                        ),
                        d_model,
                        n_heads,
                        use_fusion=True,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        final_residual, seasonal, trend = self.decomp(x)

        enc_out = self.embedding(final_residual)
        seasonal_emb = self.embedding(seasonal)

        enc_out, series, prior, sigmas = self.encoder(enc_out, seasonal_emb)

        residual_rec = self.projection(enc_out)
        final_rec = residual_rec + seasonal + trend

        if self.output_attention:
            return final_rec, final_residual, residual_rec, series, prior, sigmas

        return final_rec, final_residual, residual_rec
