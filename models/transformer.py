import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np

import math

from utils import *


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2 # 嵌入的一半维度 half，这是为了区分正弦和余弦函数。
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device) # 频率 freqs，这是一个逐渐减小的指数函数，用于确定正弦和余弦函数的变化频率。
    args = timesteps[:, None].float() * freqs[None] # 时间步骤索引与频率的乘积，用于输入正弦和余弦函数。
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # 连接正弦和余弦函数的结果，得到最终的嵌入 
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # 如果 dim 为奇数，还会添加一列零以保持偶数维度。
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h) # [128, 20, 512]
        return h

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim) # 512 -> 1024
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x)))) # torch.Size([128, 20, 512])
        y = x + self.proj_out(y, emb) # (128, 20, 512)
        return y

class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        """
        x: B, T, D
        DROP
        MASK
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1) # D拆分到多头
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H) # 计算自注意力权重，bnmh   D // H)：取 D 除以 H 的整数部分
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb) # torch.Size([128, 20, 512])
        return y # B, T, D

class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        # self.se_block =  SqueezeAndExcitationBlock(
        #     latent_dim, num_head, dropout, time_embed_dim)
        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb):
        # x = self.se_block(x) # B,T,D
        x = self.sa_block(x, emb) # B, T ,D
        x = self.ffn(x, emb) # B, T ,D
        return x

# MotionTransformer
class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads, # 3.多头数量
                    dropout=dropout,
                )
            )
        
        self.linear_projection = nn.Linear(self.latent_dim * 2, self.latent_dim)  # 假设投影到维度为 256
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats)) # 512 -> 48

    def forward(self, x, timesteps, mod=None): # xt (128,20,48) 
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        # B, D
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) # (128,512)

        # B, D
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1)) # traj_dict_mod观测DCT？  # (128,512)
            emb = emb + mod_proj

        # B, T, D
        h = self.joint_embed(x) # (128,20,512) 48 -> 512

        # 1.关闭加噪？
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :] # (128,20,512) 加噪，
        
        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
            elif i >= (self.num_layers // 2):
                h = module(h, emb)
                # 2.拼接后线性变换？
                h += prelist[-1]
                # concatenated_tensor = torch.cat((h, prelist[-1]), dim=2)
                # h = self.linear_projection(concatenated_tensor) # B, T, D
                prelist.pop()
            i += 1                                   # 最后h维度 (128,20,512)

        output = self.out(h).view(B, T, -1).contiguous() # (128,20,48) 从512 -> 48 ,确保返回张量是连续存储的
        return output

####################################################################################

# FFN
class FeedForwradNet(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb = None):
        y = self.linear2(self.dropout(self.activation(self.linear1(x)))) # torch.Size([128, 20, 512])
        # y = x + self.proj_out(y, emb) # (128, 20, 512)
        return y

# 挤压和刺激模块
class SqueezeAndExcitationBlock(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        # self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_average_pooling = nn.AvgPool1d(kernel_size = latent_dim)
        self.fc1 = zero_module(nn.Linear(1, latent_dim))
        # ReLU 激活函数
        self.relu = nn.ReLU()
        self.fc2 = zero_module(nn.Linear(latent_dim, latent_dim))
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x_se = self.global_average_pooling(x) # B,T,1
        x_se = self.relu(self.fc1(x_se)) # B,T,1
        x_se = self.sigmoid(self.fc2(x_se))
        x = x_se * x # Pointwise Multiplication
        return x

# 多头注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 注意力投影
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 注意力投影
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # 分头
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 自注意力计算
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # 合并头
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        # 输出投影
        output = self.output_projection(attention_output)

        return output

# 下面的注意力机制配套的，暂时不使用
class StylizationBlock_(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h) # [128, 20, 512]
        return h

# 多头注意力机制
class TemporalMultiHeadSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H) # 计算自注意力权重，bnmh   D // H)：取 D 除以 H 的整数部分
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        # y = x + self.proj_out(y, emb) # torch.Size([128, 20, 512]) 嵌入emb信息
        y = x + y
        return y

class SETransformerBlock(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        self.se_block = SqueezeAndExcitationBlock(
            latent_dim, num_head, dropout, time_embed_dim)
        self.mhsa = TemporalMultiHeadSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.norm = nn.LayerNorm(normalized_shape=latent_dim) # 32?
        # self.mhsa = MultiHeadSelfAttention() # Multi-head Self-attention
        self.ffn = FeedForwradNet(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb = None):
        # x_se = self.se_block(x) # torch.Size([128, 20, 512])不知道有没有emb
        # x = x + x_se
        x_mh = self.norm(x) # Norm
        # x_mh = self.mhsa(x_mh);
        x_mh = self.mhsa(x_mh)
        x = x + x_mh
        x_mh = self.norm(x) # Norm
        x_mh = self.ffn(x_mh, emb)# torch.Size([128, 20, 512])
        x = x + x_mh
        return x

# MotionTransFusion
class TransFusion(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim)) # 从标准正态分布中采样

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                SETransformerBlock(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )

        # 定义一个线性映射
        self.linear_projection = nn.Linear(2 * 512, 512)  # 假设投影到维度为 256
        # 定义线性投影层，将输入维度从 21 转换为 20
        self.linear_projection_final = nn.Linear(512, 512)
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats)) # 512 -> 48

    def forward(self, x, timesteps, mod): # xt (128,20,48) 
        """
        x: B, T, D
        timesteps: T
        mod: obverse B, 20, 48
        """
        B, T = x.shape[0], x.shape[1]

        # T, D
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        # B, 20, 48 => B, 960 => B, D
        mod_proj = self.cond_embed(mod.reshape(B, -1)) # traj_dict_mod观测DCT 
        emb = emb + mod_proj

        # B, T, 48 => B, T, D
        h = self.joint_embed(x)
        # h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :] # (128,20,512) 加入随机采样的噪声？
        
        # h前面加一个维度，连接t和c的embedding，即最后维度为 1+L
        # 在第二个维度前面添加 emb
        # B, T+1, D
        h = torch.cat((emb.unsqueeze(1), h), dim=1)
        # TODO:把t和c分开？

        i = 0
        prelist = []
        # 8层 : 4浅层 + 4深层
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                h = module(h)                
                prelist.append(h)
            elif i >= (self.num_layers // 2):
                h = module(h)
                # B, 2*(T+1), D
                concatenated_tensor = torch.cat((h, prelist[-1]), dim=2)
                # B, T+1, D
                h = self.linear_projection(concatenated_tensor)
                prelist.pop()
            i += 1

        # B, T+1, D => B, T, 48
        output = self.out(self.linear_projection_final(h[:, 1:, :])).view(B, T, -1).contiguous() # (128,20,48) 从512 -> 48 ,确保返回张量是连续存储的
        return output