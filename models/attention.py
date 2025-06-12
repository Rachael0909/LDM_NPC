import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 线性投影得到 Q, K, V
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # 重新 reshape 成多头
        q = q.view(interim_shape).transpose(1, 2)  # (Batch_Size, H, Seq_Len, Dim/H)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # 计算 attention 权重
        weight = q @ k.transpose(-1, -2)  # (Batch_Size, H, Seq_Len, Seq_Len)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        #(Batch_size, H, Seq_Len, Dim/H)-> (Batch_size, Seq_Len, H, Dim/H)
        output = output.transpose(1,2)

        output = output. reshape(input_shape)

        output = self.out_proj(output)

        #(Batch_size,Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent)  (Batch_Size, Seq_Len_Q, Dim_Q)
        # y: (context) (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 1024, 4)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # reshape q
        q = self.q_proj(x)
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2)

        # reshape k, v
        k = self.k_proj(y)
        v = self.v_proj(y)
        kv_len = y.shape[1]
        k = k.view(batch_size, kv_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.n_heads, self.d_head).transpose(1, 2)


        #attention
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output


   

