import torch
import random
from torch import nn
import os
from numpy import power
from math import ceil
from einops import rearrange

class random_masking(nn.Module):
    """Mask generator."""

    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # 4 dims
        N, num_feat, L, D = x.shape  # batch, feature, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise = noise.unsqueeze(1).repeat(1, num_feat, 1)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, num_feat, L], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return x_masked, mask, ids_restore

class PatchEmbedding(nn.Module):
    """Patchify time series."""
    def __init__(self, patch_size, in_channel, embed_dim=None, norm_layer=None, dtype=torch.float):
        super().__init__()
        self.num_patch = 1
        self.patch_size = patch_size
        # the L
        self.input_channel = in_channel
        self.embed_dim = embed_dim
        if embed_dim:
            self.input_embedding = nn.Linear(self.patch_size, self.embed_dim)

        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """
        # B, L, F
        x = torch.transpose(x, 1, 2)
        batch_size, num_feat, len_time_series = x.shape
        # number of patches
        self.num_patch = len_time_series // self.patch_size

        # Batch*num_feat, emb_d, patch_num
        patches = x.reshape(batch_size, num_feat, self.num_patch, self.patch_size)

        if self.embed_dim:
            patches = self.input_embedding(patches)
            patches = self.norm_layer(patches)
            # reshape to 4d
            patches = patches.view(batch_size, num_feat, self.num_patch, self.embed_dim)
        assert patches.shape[-2] == self.num_patch
        return patches

class PatchEmbedding_step(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(1,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.input_embedding = nn.Conv2d(1,
                                         embed_dim,
                                         kernel_size=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """
        long_term_history = torch.transpose(long_term_history, 1, 2)
        long_term_history = torch.unsqueeze(long_term_history, 2)
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        output = output.transpose(-1, -2)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, configs, patch_embedding):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, configs.enc_in, patch_embedding.num_patch, configs.d_model))

    def forward(self, input_data, index=None, abs_idx=None):
        batch_size, num_nodes, num_patches, embed_dim = input_data.shape
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)  # .unsqueeze(0)
        input_data = input_data + pe

        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, embed_dim)

        return input_data

# Dimension invariant embedding
class DI_embedding(nn.Module):
    def __init__(self, seg_len, embed_dim, dropout):
        super(DI_embedding, self).__init__()
        self.convs = nn.ModuleList()
        self.seg_len = seg_len
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=embed_dim,
                                kernel_size=(3, 1),
                                padding='same')
        self.norm = nn.LayerNorm(embed_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        return x

class TS_Segment(nn.Module):
    def __init__(self, seq_len, seg_len):
        super(TS_Segment, self).__init__()
        self.seg_len = seg_len
        self.seg_num = ceil(seq_len / self.seg_len)
        self.pad_len = self.seg_num * self.seg_len - seq_len

    def concat(self, x):
        # The shape of x:[batch_size, d_model, seg_num, seg_len, feature dims]
        batch_size, emb_d, seg_num, seg_len, ts_d = x.shape
        x = rearrange(x, 'b d_model seg_num seg_len ts_d -> b d_model (seg_num seg_len) ts_d')
        if self.pad_len != 0:
            x = x[:, :, :(seg_num*seg_len-self.pad_len), :]
        return x

    def forward(self, x):
        # The shape of x:[batch_size, d_model, seq_len, feature dims]
        batch_size, emb_d, ts_len, ts_d = x.shape
        if self.pad_len != 0:
            x = torch.cat([x, torch.zeros(batch_size, emb_d, self.pad_len, ts_d, device=x.device)], dim=2)
        # conduct segmentation to time series data to the time step dimension.
        x_segment = rearrange(x,
                              'b d_model (seg_num seg_len) ts_d -> b d_model seg_num seg_len ts_d',
                              seg_len=self.seg_len)
        return x_segment


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=0)
    # find period by amplitudes
    frequency_list = abs(xf).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[0] // top_list
    period = period[period < 200]
    return period, abs(xf).mean(-1)[top_list]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean