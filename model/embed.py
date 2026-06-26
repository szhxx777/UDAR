import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class UncertaintyWeightedMovingAvg(nn.Module):
    """
    Uncertainty-weighted moving average used for UDAR decomposition.
    """

    def __init__(self, kernel_size, stride=1):
        super(UncertaintyWeightedMovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.padding_size = (kernel_size - 1) // 2

        self.trend_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                                    stride=stride, padding=0, bias=False)
        nn.init.constant_(self.trend_conv.weight, 1.0 / kernel_size)

        self.uncertainty_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        self.weight_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, D = x.size()

        frontend = x[:, 0:1, :].repeat(1, self.padding_size, 1)
        backend = x[:, -1:, :].repeat(1, self.padding_size, 1)
        x_padded = torch.cat([frontend, x, backend], dim=1)

        x_reshaped = x_padded.permute(0, 2, 1).contiguous().view(B * D, 1, -1)

        trend_raw = self.trend_conv(x_reshaped)

        x_squared = x_reshaped ** 2
        mu = self.uncertainty_pool(x_reshaped)
        mu_sq = self.uncertainty_pool(x_squared)

        var = mu_sq - mu ** 2
        var = torch.relu(var)
        std = torch.sqrt(var + 1e-6)

        std_flat = std.permute(0, 2, 1).contiguous().view(-1, 1)
        weights = self.weight_mlp(std_flat)

        trend_raw = trend_raw.view(B, D, L).permute(0, 2, 1)
        weights = weights.view(B, D, L).permute(0, 2, 1)

        trend_final = trend_raw * weights

        return trend_final


class UncertaintyDecomposition(nn.Module):
    """
    Residual-seasonal-trend decomposition with uncertainty-aware smoothing.
    """

    def __init__(self, trend_kernel_size, seasonal_kernel_size):
        super(UncertaintyDecomposition, self).__init__()
        self.trend_avg = UncertaintyWeightedMovingAvg(trend_kernel_size, stride=1)
        self.seasonal_avg = UncertaintyWeightedMovingAvg(seasonal_kernel_size, stride=1)

    def forward(self, x):
        trend = self.trend_avg(x)
        remainder_sr = x - trend

        seasonal = self.seasonal_avg(remainder_sr)

        final_residual = remainder_sr - seasonal

        return final_residual, seasonal, trend

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class MovingAvg(nn.Module):
    """Simple moving average helper."""
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class StandardDecomposition(nn.Module):
    """
    Standard residual-seasonal-trend decomposition.
    """
    def __init__(self, trend_kernel_size, seasonal_kernel_size):
        super(StandardDecomposition, self).__init__()
        self.trend_avg = MovingAvg(trend_kernel_size, stride=1)
        self.seasonal_avg = MovingAvg(seasonal_kernel_size, stride=1)

    def forward(self, x):
        trend = self.trend_avg(x)
        remainder_sr = x - trend
        seasonal = self.seasonal_avg(remainder_sr)
        final_residual = remainder_sr - seasonal
        return final_residual, seasonal, trend
