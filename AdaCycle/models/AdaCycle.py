import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Cycle_Block import CycleBlock
from layers.revin import RevIN
from layers.common import RecurrentCycle, QKCompressedAttention, ResAttention


class Model(nn.Module):
    """
    AdaCycle: Adaptive Multi-Cycle Fusion with Heterogeneous Wavelet Decomposition for Time Series Forecasting
    """

    def __init__(self, configs, cycle_inf):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.top_k = len(cycle_inf[0])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_revin = configs.use_revin
        self.m = configs.m

        # Detected cycle periods and weights
        self.cycle_list = cycle_inf[0]
        self.cycle_weight = cycle_inf[1]

        # Feature dimension after multi-cycle concatenation
        self.fwv = configs.d_model * self.top_k

        # Reversible normalization
        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)

        # Learnable cycle patterns
        self.cycleQueue = RecurrentCycle(self.cycle_list, self.enc_in, self.device)

        # Wavelet decomposition and heterogeneous modeling
        self.CycleBlock = CycleBlock(configs, self.top_k)

        self.linear = nn.ModuleList(nn.Conv1d(configs.seq_len, configs.d_model,1)
                                    for _ in range(self.top_k))

        # Cross-cycle attention with QK compression
        self.cross_attention = QKCompressedAttention(
            ResAttention(configs.attn_dropout),
            self.fwv,
            self.top_k,
            configs.n_heads
        )

        # Feed-forward network
        d_ff = configs.d_ff if configs.d_ff is not None else self.fwv * 2
        self.conv1 = nn.Conv1d(self.fwv, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, self.fwv, 1)
        self.norm1 = nn.LayerNorm(self.fwv)
        self.norm2 = nn.LayerNorm(self.fwv)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.proj = nn.Linear(self.fwv, self.pred_len)

    def forward(self, x_enc, x_mark_enc, cycle_index):
        """
        Args:
            x_enc: [batch, seq_len, channels]
            x_mark_enc: [batch, seq_len, mark_dim]
            cycle_index: [batch, top_k]
        Returns:
            dec_out: [batch, pred_len, channels]
        """

        # Reversible normalization
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, 'norm')

        # Multi-cycle independent processing
        dec_list = []
        for i in range(len(self.cycle_list)):
            # Detrending: remove cycle component
            cycle_component = self.cycleQueue(cycle_index[:, i], self.seq_len, i)
            x_detrend = x_enc - cycle_component

            # Wavelet decomposition + heterogeneous modeling
            cycle_output = self.CycleBlock(x_detrend, x_mark_enc, i)

            # add cycle component
            dec_list.append(cycle_output[:,:self.enc_in,:] + self.linear[i](cycle_component).permute(0,2,1))

        # Multi-cycle fusion with weights
        dec_out = torch.stack(dec_list, dim=-1)  # [batch, channels, d_model, top_k]
        B, C, D, P = dec_out.shape

        # Adaptive weighted fusion
        weight = self.cycle_weight.to(dec_out.device)
        weight = F.softmax(weight, dim=-1)
        weight = weight.view(1, 1, 1, -1)

        # Concatenate weighted cycle features
        dec_out = (dec_out * weight).permute(0, 1, 3, 2).reshape(B, C, -1)

        # Cross-cycle attention aggregation
        attn_out, _ = self.cross_attention(dec_out, dec_out, dec_out)
        dec_out = dec_out + attn_out

        # Feed-forward network
        x_norm = self.norm1(dec_out)
        y = self.dropout(self.activation(self.conv1(x_norm.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(y + x_norm)

        # Temporal projection
        y = self.proj(y)  # [batch, channels, pred_len]
        dec_out = y.permute(0, 2, 1)[:, :, :self.enc_in]

        # Reversible denormalization
        if self.use_revin:
            dec_out = self.revin_layer(dec_out, 'denorm')

        return dec_out