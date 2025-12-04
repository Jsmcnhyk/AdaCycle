import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.common import WaveletEmbedding


class FeatureDimRecalibration(nn.Module):
    """
    Feature dimension recalibration with channel independence
    """

    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.SiLU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, channels, d_model]
        scale = self.fc(x)
        return x * scale


class LearnableScaling(nn.Module):
    """
    Learnable scaling factor for each feature dimension
    """

    def __init__(self, d_model, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, d_model) * init_value)

    def forward(self, x):
        # x: [batch, channels, d_model]
        return x * self.scale


class DualSwiGLU_WithRecalibrationScaling(nn.Module):
    """
    Dual-layer SwiGLU with feature recalibration and scaling
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.gate1 = nn.Linear(d_model, d_model)
        self.value1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.recalibration = FeatureDimRecalibration(d_model, 4)
        self.scaling1 = LearnableScaling(d_model, 1.0)
        self.scaling2 = LearnableScaling(d_model, 1.0)

        self.drop_path = nn.Dropout(0.2)

        self.gate2 = nn.Linear(d_model, d_model)
        self.value2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        r = x
        gate1 = F.silu(self.gate1(x))  # SiLU激活
        x = gate1 * self.value1(x)

        x = self.recalibration(x + r)
        x = self.scaling1(x)
        x = self.drop_path(x)

        gate2 = F.silu(self.gate2(x))
        x = gate2 * self.value2(x)
        x = self.scaling2(x)
        return x


class CycleBlock(nn.Module):
    """
    Cycle-specific processing with wavelet decomposition and heterogeneous modeling
    """

    def __init__(self, configs, top_k):
        super(CycleBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.top_k = top_k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.d_model = configs.d_model
        self.m = configs.m
        self.fwv = self.d_model * (self.m + 1)

        # Temporal mark dimension
        self.mark_c = 0
        if configs.data in ['ETTh1', 'ETTh2', 'custom', 'ETTm1', 'ETTm2']:
            self.mark_c = 4
        elif configs.data in ['Solar', 'PEMS']:
            self.mark_c = 0

        # Shared wavelet transform (stationary wavelet transform)
        self.swt = WaveletEmbedding(
            self.enc_in + self.mark_c,
            True,
            True,
            configs.wv,
            self.m,
            configs.kernel_size
        )
        self.dswt = WaveletEmbedding(
            self.enc_in + self.mark_c,
            False,
            True,
            configs.wv,
            self.m,
            configs.kernel_size
        )

        # Input projection: seq_len -> d_model
        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.seq_len, self.d_model)
            for _ in range(self.top_k)
        ])

        # Low frequency processor (linear for trend)
        self.low_freq_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.AvgPool1d(2),
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                LearnableScaling(self.d_model, 1.0)
            ) for _ in range(self.top_k)
        ])

        # High frequency processor (non-linear for patterns)
        self.high_freq_processor = nn.ModuleList([
            DualSwiGLU_WithRecalibrationScaling(self.d_model, configs.dropout)
            for _ in range(self.m * self.top_k)
        ])

        # Output projection: d_model -> pred_len
        self.linear_list2 = nn.ModuleList([
            nn.Linear(self.d_model, self.pred_len)
            for _ in range(self.top_k)
        ])

    def forward(self, x, x_mark_c, i):
        """
        Args:
            x: [batch, seq_len, channels]
            x_mark_c: [batch, seq_len, mark_dim] or None
            i: cycle index
        Returns:
            x_rec: [batch, channels, d_model]
        """

        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]

        # Temporal projection with optional marks
        if x_mark_c is not None:
            x = torch.cat([x, x_mark_c.permute(0, 2, 1)], dim=1)
        x = self.linear_list1[i](x)  # [batch, channels+marks, d_model]

        # Wavelet decomposition
        x_w = self.swt(x)  # [batch, channels, num_wavelets, d_model]
        B, C, W, D = x_w.shape

        # Low frequency: linear modeling
        ll_feat = self.low_freq_processor[i](x_w[:, :, 0, :])
        x_wl = ll_feat

        # High frequency: non-linear modeling
        for j in range(self.m):
            lh_feat = self.high_freq_processor[i * self.m + j](x_w[:, :, j + 1, :])
            x_wl = torch.cat((x_wl, lh_feat), dim=-1)

        # Reshape for wavelet reconstruction
        x_w_processed = x_wl.reshape(B, C, self.d_model, self.m + 1).permute(0, 1, 3, 2)

        # Wavelet reconstruction
        x_rec = self.dswt(x_w_processed)

        return x_rec