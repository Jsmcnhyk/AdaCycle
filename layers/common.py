import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import pywt
import matplotlib.pyplot as plt



class RecurrentCycle(torch.nn.Module):

    def __init__(self, cycle_list, channel_size, device):
        super(RecurrentCycle, self).__init__()
        self.cycle_list = cycle_list
        self.channel_size = channel_size

        self.data_list = nn.ParameterList([
            nn.Parameter(torch.zeros(int(cycle), channel_size, device=device))
            for cycle in cycle_list
        ])

    def forward(self, index, length, i):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_list[i]
        return self.data_list[i][gather_index]




class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)

            #  'db2'-->kernel_size = 4
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)


            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:

            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)   # h0低通滤波器
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)   # h1高通滤波器
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)


    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")

            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])  #  (out_channels, in_channels/groups, kernel_size)
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2

        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff


class QKCompressedAttention(nn.Module):
    def __init__(self, attention, d_model, compression, n_heads):
        super(QKCompressedAttention, self).__init__()

        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model//compression)
        self.k_proj = nn.Linear(d_model, d_model//compression)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q_proj(q).reshape(B, L, H, -1)
        k = self.k_proj(k).reshape(B, S, H, -1)
        v = self.v_proj(v).reshape(B, S, H, -1)

        out, attn = self.attention(
            q, k, v,
            res=res, attn=attn
        )
        out = out.view(B, L, -1)

        return self.out(out), attn


class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None, attn_map=False, nst=False):
        super(ResAttention, self).__init__()

        self.nst = nst
        self.scale = scale
        self.attn_map = attn_map
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        if self.attn_map is True:
            heat_map = attn_map.reshape(32, -1, H, L, S)
            for b in range(heat_map.shape[0]):
                for c in range(heat_map.shape[1]):
                    h_map = heat_map[b, c, 0, ...].detach().cpu().numpy()
                    # plt.savefig(heat_map, f'{b} sample {c} channel')

                    plt.figure(figsize=(10, 8), dpi=200)
                    plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                    plt.colorbar()

                    plt.rcParams['font.family'] = 'serif'
                    plt.rcParams['font.serif'] = ['Times New Roman']
                    plt.xlabel('Key Time Patch', fontsize=14)
                    plt.ylabel('Query Time Patch', fontsize=14)
                    plt.tight_layout()
                    if self.nst is True:
                        plt.savefig(f'./time map/{b}_sample_{c}_channel.png')
                    else:
                        plt.savefig(f'./stable time map/{b}_sample_{c}_channel.png')

                    plt.close()
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A




