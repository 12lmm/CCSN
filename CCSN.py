
import torch
import numpy as np
from torch import isin, nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
from torch.nn.modules.batchnorm import _BatchNorm

import os
import math
import random

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD # 用于图像数据预处理
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, LayerNorm2d
from timm.layers.norm import _is_contiguous
from timm.models._registry import register_model  # 与注册深度学习模型的操作有关，允许用户将自定义的模型添加到库中


from mmengine.registry import MODELS
from mmengine.model import constant_init, kaiming_init
from mmengine.runner import load_checkpoint

from mmcv.cnn import (
    build_norm_layer,
    build_activation_layer,
)

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 检查是否成功安装xformers.ops
try:
    import xformers.ops as xops

    has_xformers = True
except ImportError:
    has_xformers = False


from .layers.blur_pool import BlurConv2d
from .layers.downsample import build_downsample_layer


__all__ = ["CCSN"]

# 重新排列input tensor的维度（通道优先的内存格式，用于后续的深度学习框架）
def c_rearrange(x, H, W, dim=1):
    channels_last = x.is_contiguous(memory_format=torch.channels_last)
    if dim == 1:
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
    elif dim == 2:
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
    else:
        raise NotImplementedError

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)  # 检查输入张量是否是以通道优先的内存格式
    else:
        x = x.contiguous()                                   # 如果不是，就调用contiguous，确保是连续存储的
    return x

# 深度可分离卷积
class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=True, with_shortcut=False):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, 1, (kernel_size - 1) // 2, bias=bias, groups=dim
        )
        self.with_shortcut = with_shortcut   # 存储对象的状态或配置信息

    def forward(self, x, H, W):
        shortcut = x
        x = c_rearrange(x, H, W)
        x = self.dwconv(x)
        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W)
        if self.with_shortcut:
            return x + shortcut
        return x

# MLP网络 + 可选的DWConv操作
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        with_dwconv=False,
        with_shortcut=False,
        act_cfg=dict(type="GELU"),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = (                                      # 一个可选的DWConv操作
            DWConv(hidden_features, with_shortcut=with_shortcut)
            if with_dwconv
            else None
        )
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.dwconv is not None:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 残差连接，其中的fn是什么函数
class Residual(nn.Module):
    def __init__(self, fn, drop_path_rate=0.0):
        super().__init__()
        self.fn = fn

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        return self.drop_path(self.fn(x)) + x

###### 将输入 映射到 聚类空间
  # 将input tensor 用线性变换（减维度dim—>num_clusters），并重排序
class P2CLinear(nn.Module):
    def __init__(self, dim, num_clusters, **kwargs) -> None:
        super().__init__()

        self.clustering = nn.Sequential(
            nn.Linear(dim, num_clusters, bias=False),
            Rearrange("B N M -> B M N"),
        )

    def forward(self, x, H, W):
        return self.clustering(x)

  # 将input tensor 用MLP中的一层 B N M（批次，通道数，图高宽） -> B M N（批次，图高宽，聚类数目）
class P2CMlp(nn.Module):
    def __init__(
        self, dim, num_clusters, mlp_ratio=4.0, act_cfg=dict(type="GELU"), **kwargs
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)         # 计算隐藏层的维度
        self.clustering = nn.Sequential(          # MLP的一个隐藏层
            nn.Linear(dim, hidden_dim),
            build_activation_layer(act_cfg),
            nn.Linear(hidden_dim, num_clusters),  # TODO: train w/ bias=False
            Rearrange("B N M -> B M N"),
        )

    def forward(self, x, H, W):
        return self.clustering(x)



  # 将input tensor 用卷积 映射到一个 维度较小 的表示
  #  kernel_size=7
class P2CConv2d(nn.Module):
    def __init__(
        self,
        dim,
        num_clusters,
        kernel_size=7,
        act_cfg=dict(type="GELU"),
        **kwargs,
    ) -> None:
        super().__init__()

        self.clustering = nn.Sequential(         # 深度可分离卷积 kernel_size=7
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=dim,
            ),
            build_activation_layer(act_cfg),
            nn.Conv2d(dim, dim, 1, 1, 0),
            build_activation_layer(act_cfg),
            nn.Conv2d(dim, num_clusters, 1, 1, 0, bias=False),
            Rearrange("B M H W -> B M (H W)"),
        )

    def forward(self, x, H, W):
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        # print("P2C -> x shape:", x.shape)
        return self.clustering(x)



class CyrusSOM(object):
    def __init__(
            self,
            net = [1] * 50,
            epochs=2,
            r_t=[None, None],
            eps=1e-6):
        """
        :param net: 竞争层的拓扑结构，支持一维及二维，1表示该输出节点存在，0表示不存在该输出节点
        :param epochs: 最大迭代次数
        :param r_t: [C,B] 领域半径参数，r = C*e**(-B*t/eoochs),其中t表示当前迭代次数
        :param eps: learning rate的阈值
        """
        self.epochs = epochs
        self.C = r_t[0]
        self.B = r_t[1]
        self.eps = eps
        self.output_net = np.array(net)
        if len(self.output_net.shape) == 1:
            self.output_net = self.output_net.reshape([-1, 1])
        self.coord = np.zeros([self.output_net.shape[0], self.output_net.shape[1], 2])
        for i in range(self.output_net.shape[0]):
            for j in range(self.output_net.shape[1]):
                self.coord[i, j] = [i, j]

    def __r_t(self, t):
        if not self.C:
            return 0.5
        else:
            return self.C * np.exp(-self.B * t / self.epochs)

    def __lr(self, t, distance):
        return (self.epochs - t) / self.epochs * np.exp(-distance)

    def standard_x(self, x):
        # x = rearrange(x, "B C H W -> B C (H W)")
        d = x[:, :, 0, 0]
        x = d

        x = np.array(x.detach().cpu())
        for i in range(x.shape[0]):
            x[i, :] = [value / (((x[i, :]) ** 2).sum() ** 0.5) for value in x[i, :]]
        return x

    def standard_w(self, w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i, j, :] = [value / (((w[i, j, :]) ** 2).sum() ** 0.5) for value in w[i, j, :]]
        return w

    def cal_similar(self, x, w):
        similar = (x * w).sum(axis=2)
        coord = np.where(similar == similar.max())
        return [coord[0][0], coord[1][0]]

    def update_w(self, center_coord, x, step):
        for i in range(self.coord.shape[0]):
            for j in range(self.coord.shape[1]):
                distance = (((center_coord - self.coord[i, j]) ** 2).sum()) ** 0.5
                if distance <= self.__r_t(step):
                    self.W[i, j] = self.W[i, j] + self.__lr(step, distance) * (x - self.W[i, j])

    def forward(self, x):
        # print("forward,x shape:", x.shape)
        self.train_x = self.standard_x(x)
        # print("output_net shape:", self.output_net.shape)
        # print("train_x shape:", self.train_x.shape)
        # print("self.train_x type:", type(self.train_x))
        self.W = np.zeros([self.output_net.shape[0], self.output_net.shape[1], self.train_x.shape[1]])
        # print("W shape:", self.W.shape)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j, :] = self.train_x[random.choice(range(self.train_x.shape[0])), :]

        self.W = self.standard_w(self.W)
        for step in range(int(self.epochs)):
            j = 0
            if self.__lr(step, 0) <= self.eps:
                break
            for index in range(self.train_x.shape[0]):
                center_coord = self.cal_similar(self.train_x[index, :], self.W)
                self.update_w(center_coord, self.train_x[index, :], step)
                self.W = self.standard_w(self.W)
                j += 1
        label = []  # label是一个包含每个输入样本所属的聚类标签的列表,一维数组
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[0])
        class_dict = {}
        for index in range(self.train_x.shape[0]):
            if label[index] in class_dict.keys():
                class_dict[label[index]].append(index)
            else:
                class_dict[label[index]] = [index]

        # print("class_dict, x shape:", x.shape)
        B = x.shape[0]
        M = 100
        H = x.shape[2]
        W = x.shape[3]

        d = x[:, :, 0, 0]
        x = d
        x = np.array(x.detach().cpu())
        # print("afetr class_dict, x shape:", x.shape)  # x: torch.Size([2, 96, 56, 56])-> (2, 96)
        # cluste center
        cluster_center = {}
        for key, value in class_dict.items():
            cluster_center[key] = np.array([x[i, :] for i in value]).mean(axis=0)
        self.cluster_center = cluster_center
        # print(f"cluster_center: {cluster_center}")

        clustered_features = torch.zeros(B, M, H, W)
        # print("clustered_features of shape:", clustered_features.shape)

        # 将每个聚类中心填充到对应位置
        for key, value in cluster_center.items():
            # print(f"Key: {key}, Value: {value}")
            # print(f"Cluster {key} center shape:", value.shape)
            # 将 NumPy 数组转换为张量，并添加扩展维度使其与目标形状匹配
            value_tensor = torch.tensor(value).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # print(f"=======Cluster {key} center shape:", value_tensor.shape)
            m = value_tensor.shape[1]
            value = value_tensor.expand(B, m, H, W)
            # print(f"Value tensor shape before expansion: {value_tensor.shape}")
            # print(f"Value shape : {value.shape}")

            # print(f"=========Cluster center shape:", value.shape)
            # print(f"=========clustered_features {key}  shape:", clustered_features[:, key, :, :].shape)

            # 将聚类中心填充到对应位置
            clustered_features[:, key, :, :] = value[:, key, :, :]
        clustered_features = clustered_features.view(B, M, H * W)
        # print("================clustered_features of shape:", clustered_features.shape)

        # 返回聚类后的特征张量
        return clustered_features

    def __call__(self, x, H, W):
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        # print("x shape:", x.shape)
        return self.forward(x)

###### 进行聚类注意力计算的层
  # Patch-to-Cluster Attention Layer
class PaCaLayer(nn.Module):
    """Patch-to-Cluster Attention Layer"""

    def __init__(
        self,
        paca_cfg,
        dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_ratio=4.0,
        act_cfg=dict(type="GELU"),
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads

        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.num_clusters = paca_cfg["clusters"]
        self.onsite_clustering = paca_cfg["onsite_clustering"]
        if self.num_clusters > 0:                                     # 聚类结果通过指定的归一化方式进行归一化
            self.cluster_norm = (                                     # 创建一个用于归一化聚类结果的归一化层
                build_norm_layer(paca_cfg["cluster_norm_cfg"], dim)[1]
                if paca_cfg["cluster_norm_cfg"]
                else nn.Identity()
            )

            if paca_cfg["type"] == "CyrusSOM":
                self.clustering = eval(paca_cfg["type"])(
                    # num_clusters=self.num_clusters,
                    # mlp_ratio=mlp_ratio,
                    # kernel_size=paca_cfg["clustering_kernel_size"],
                    # act_cfg=act_cfg,
                )
            else:
                self.clustering = eval(paca_cfg["type"])(
                    dim=dim,
                    num_clusters=self.num_clusters,
                    mlp_ratio=mlp_ratio,
                    kernel_size=paca_cfg["clustering_kernel_size"],
                    act_cfg=act_cfg,
                )

            # if self.onsite_clustering:                                   # 动态选择聚类器的类型
            #     self.clustering = eval(paca_cfg["type"])(
            #         dim=dim,
            #         num_clusters=self.num_clusters,
            #         mlp_ratio=mlp_ratio,
            #         kernel_size=paca_cfg["clustering_kernel_size"],
            #         act_cfg=act_cfg,
            #     )

            self.cluster_pos_embed = paca_cfg["cluster_pos_embed"]
            if self.cluster_pos_embed:
                self.cluster_pos_enc = nn.Parameter(
                    torch.zeros(1, self.num_clusters, dim)
                )
                trunc_normal_(self.cluster_pos_enc, std=0.02)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity()                                      # 获取注意力权重以进行可视化的标识层

    def forward(self, x, H, W, z, alpha1=0.2, weight_factor=2.0): # (alpha1的取值：0.2，0.4，0.6，0.8)
                                                                #  (alpha2的取值：0.1，0.2，0.3，0.4)
        # x: B N C

        #### 计算z
        if self.num_clusters > 0:
            if self.onsite_clustering:
                z_raw = self.clustering(x, H, W)  # B M N                 # 计算聚类结果z
                z = z_raw.softmax(dim=-1)                                 # 在最后一个维度上进行 softmax 操作


                # TODO: how to auto-select the 'meaningful' subset of clusters
            # 获取每个样本的第一和第二高置信度分数
            top2_conf, top2_indices = torch.topk(z, k=2, dim=-1)  # 获取前两个最大值及其索引
            top1_conf = top2_conf[:, :, 0]  # 第一置信度分数 (λ₁)
            top2_conf_val = top2_conf[:, :, 1]  # 第二置信度分数 (λ₂)

            # 计算第一置信度和第二置信度的差值
            conf_diff = top1_conf - top2_conf_val  # 置信度差值 (λ₁ - λ₂)

            #
            alpha2 = alpha1 * 0.5


            adaptive_factor = torch.mean(conf_diff) * 0.5
            weight_factor = adaptive_factor + weight_factor

            # 选择满足可靠性条件的样本：λ₁ >= α₁ 且 (λ₁ - λ₂) >= α₂
            reliable_samples = (top1_conf >= alpha1) & (conf_diff >= alpha2)

            # 生成权重向量：可靠样本权重大，不可靠样本权重小
            weight = reliable_samples.float() * weight_factor + (1 - reliable_samples.float())

            # c = z @ x  # B M C
            c2 = einsum("bmn,bnc->bmc", z, x)
            # 对聚类特征c加权
            c_weighted = c2 * weight.unsqueeze(-1)  # 对聚类特征进行加权

            if self.cluster_pos_embed:
                c_weighted = c_weighted  + self.cluster_pos_enc.expand(c_weighted.shape[0], -1, -1)
            c = self.cluster_norm(c_weighted)                     # 生成加权后的聚类特征

        else:
            c = x
            weight = torch.ones_like(x)  # 如果没有聚类，所有样本的权重为 1


        #### 执行多头自注意力操作
        # 用xformers
        if self.use_xformers:
            # 计算自注意力，并用采样后的数据进行处理
            q = self.q(x)
            k = self.k(c)
            v = self.v(c)
            q = rearrange(q, "B N (h d) -> B N h d", h=self.num_heads)
            k = rearrange(k, "B M (h d) -> B M h d", h=self.num_heads)
            v = rearrange(v, "B M (h d) -> B M h d", h=self.num_heads)

            x = xops.memory_efficient_attention(q, k, v)  # B N h d     # 进行注意力计算
            x = rearrange(x, "B N h d -> B N (h d)")

            x = self.proj(x)
        # 用multi_head_attention_forward
        else:
            x = rearrange(x, "B N C -> N B C")
            c = rearrange(c, "B M C -> M B C")

            x, attn = F.multi_head_attention_forward(
                query=x,
                key=c,
                value=c,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q.weight,
                k_proj_weight=self.k.weight,
                v_proj_weight=self.v.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=self.attn_drop,
                out_proj_weight=self.proj.weight,
                out_proj_bias=self.proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=not self.training,  # for visualization
                average_attn_weights=False,
            )

            x = rearrange(x, "N B C -> B N C")  # x变回了原来的形状

            if not self.training:
                attn = self.attn_viz(attn)      # 如果不在训练阶段，会对注意力权重 attn 进行可视化，热图

        x = self.proj_drop(x)

        return x, z


###### 模型的基本构建块
  # 基本构建块：PaCaLayer、mlp、pos_embed
class PaCaBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        paca_cfg,
        mlp_ratio=4.0,
        drop_path=0.0,
        attn_drop=0.0,
        drop=0.0,
        act_cfg=dict(type="GELU"),
        layer_scale=None,
        input_resolution=None,
        with_pos_embed=False,
        post_norm=False,
        sub_ln=False,  # https://arxiv.org/abs/2210.06423
        **kwargs,
    ):
        super().__init__()

        self.post_norm = post_norm

        self.with_pos_embed = with_pos_embed
        self.input_resolution = input_resolution
        if self.with_pos_embed:                             # 位置嵌入
            assert self.input_resolution is not None
            self.input_resolution = to_2tuple(self.input_resolution)   # 确保input的格式是（h,w）
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.input_resolution[0] * self.input_resolution[1], dim)
            )                                                          # 创建一个可学习的位置嵌入参数pos_embed
            self.pos_drop = nn.Dropout(p=drop)
            trunc_normal_(self.pos_embed, std=0.02)                    # 一种初始化策略

        self.norm1_before = (                               # 归一化层
            build_norm_layer(paca_cfg["norm_cfg1"], dim)[1]
            if sub_ln or not post_norm                                 # sub_ln 为 True 或者 post_norm 为 False 用这个层
            else nn.Identity()
        )
        self.attn = PaCaLayer(                              # 计算attention
            paca_cfg=paca_cfg,
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            mlp_ratio=mlp_ratio,
            act_cfg=act_cfg,
        )
        self.norm1_after = (
            build_norm_layer(paca_cfg["norm_cfg1"], dim)[1]
            if sub_ln or post_norm
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2_before = (
            build_norm_layer(paca_cfg["norm_cfg2"], dim)[1]
            if sub_ln or not post_norm
            else nn.Identity()
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = eval(paca_cfg["mlp_func"])(              # 创建一个MLP模块
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            with_dwconv=paca_cfg["with_dwconv_in_mlp"],                  # 布尔值，是否用with_dwconv_in_mlp
            with_shortcut=paca_cfg["with_shortcut_in_mlp"],              # 同上
            act_cfg=act_cfg,
        )
        self.norm2_after = (
            build_norm_layer(paca_cfg["norm_cfg2"], dim)[1]
            if sub_ln or post_norm
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_scale = False                           # 默认不使用layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True  # gamma1、gamma2 用于调整两个位置的输出
            )

    def forward(self, x, H, W, z):
        # x: B N C
        if self.with_pos_embed:                          # 位置嵌入
            if self.input_resolution != (H, W):
                pos_embed = rearrange(self.pos_embed, "B (H W) C -> B C H W")
                pos_embed = F.interpolate(
                    pos_embed, size=(H, W), mode="bilinear", align_corners=True
                )
                pos_embed = rearrange(pos_embed, "B C H W -> B (H W) C")
            else:
                pos_embed = self.pos_embed

            x = self.pos_drop(x + pos_embed)

        a, z = self.attn(self.norm1_before(x), H, W, z)   # 注意力机制
        a = self.norm1_after(a)
        if not self.layer_scale:                          # 通过不同的路径计算 x 的更新，1、layer_scale=false
            x = x + self.drop_path1(a)
            x = x + self.drop_path2(
                self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )
        else:                                             # 2、layer_scale=true,gamma1、gamma2是可学习的参数
            x = x + self.drop_path1(self.gamma1 * a)
            x = x + self.drop_path2(
                self.gamma2 * self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )

        return x, z

class PaCaTeacher_ConvMixer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        stem_cfg=None,
        embed_dims=[96, 192, 320, 384],
        clusters=[100, 100, 100, 100],
        depths=[2, 2, 2, 2],
        kernel_size=7,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="GELU"),
        drop_path_rate=0.0,
        return_outs=True,
    ) -> None:
        super().__init__()

        self.num_stages = len(depths)
        self.num_clusters = clusters
        self.return_outs = return_outs

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        self.stem = None
        if stem_cfg is not None:
            stem_cfg.update(dict(in_chs=in_chans, out_chs=embed_dims[0]))
            self.stem = build_stem_layer(stem_cfg)

        for i in range(self.num_stages):
            dim = embed_dims[i]

            block = nn.Sequential(
                *[
                    nn.Sequential(
                        Residual(
                            nn.Sequential(
                                nn.Conv2d(
                                    dim, dim, kernel_size, groups=dim, padding="same"
                                ),
                                build_activation_layer(act_cfg),
                                build_norm_layer(norm_cfg, dim)[1],
                            ),
                            drop_path_rate=dpr[cur + j],
                        ),
                        nn.Conv2d(dim, dim, kernel_size=1),
                        build_activation_layer(act_cfg),
                        build_norm_layer(norm_cfg, dim)[1],
                    )
                    for j in range(depths[i])
                ]
            )
            setattr(self, f"block{i+1}", block)

            if i < self.num_stages - 1:
                transition = nn.Sequential(
                    BlurConv2d(dim, embed_dims[i + 1], 3, 2, 1),
                    build_activation_layer(act_cfg),
                    build_norm_layer(norm_cfg, embed_dims[i + 1])[1],
                )
                setattr(self, f"transition{i+1}", transition)

                lateral = nn.Sequential(
                    nn.Conv2d(embed_dims[i + 1], dim, kernel_size=1),
                    build_activation_layer(act_cfg),
                    build_norm_layer(norm_cfg, dim)[1],
                )
                setattr(self, f"lateral{i+1}", lateral)

                fpn = nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            build_activation_layer(act_cfg),
                            build_norm_layer(norm_cfg, dim)[1],
                        ),
                        drop_path_rate=dpr[cur],
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    build_activation_layer(act_cfg),
                    build_norm_layer(norm_cfg, dim)[1],
                )
                setattr(self, f"fpn{i+1}", fpn)

            to_clusters = (
                nn.Conv2d(embed_dims[i], clusters[i], 1, 1, 0)
                if clusters[i] > 0
                else None
            )
            setattr(self, f"to_clusters{i+1}", to_clusters)

            cur += depths[i]

    def forward(self, x):
        # x: B C H W
        if self.stem is not None:
            x = self.stem(x)

        outs = []
        for i in range(self.num_stages):
            block = getattr(self, f"block{i+1}")
            x = block(x)
            outs.append(x)
            if i < self.num_stages - 1:
                transition = getattr(self, f"transition{i+1}")
                x = transition(x)

        fpn_outs = [None] * len(outs)
        fpn_outs[-1] = outs[-1]
        for i in range(self.num_stages - 1, 0, -1):
            out = F.interpolate(
                fpn_outs[i], outs[i - 1].shape[2:], mode="bilinear", align_corners=False
            )
            lateral = getattr(self, f"lateral{i}")
            out = lateral(out)
            out = outs[i - 1] + out
            fpn = getattr(self, f"fpn{i}")
            fpn_outs[i - 1] = fpn(out)

        clusters = []
        for i in range(self.num_stages):
            to_clusters = getattr(self, f"to_clusters{i+1}")
            if to_clusters is not None:
                clusters.append(to_clusters(fpn_outs[i]))

        for i in range(len(clusters)):
            clusters[i] = rearrange(clusters[i], "B M H W -> B M (H W)").contiguous()
            clusters[i] = clusters[i].softmax(dim=-1)

        if self.return_outs:
            for i in range(len(outs)):
                outs[i] = rearrange(outs[i], "B C H W -> B (H W) C").contiguous()

            return clusters, outs

        return clusters, None


class CCSN(nn.Module):


    def __init__(
        self,
        in_chans=3,
        num_classes=101,
        img_size=224,  # for cls only
        # # 原：1：2个卷积，2、3、4：1个卷积 k=3,s=2,p=1
        # stem_cfg=dict(
        #     type="DownsampleV1",
        #     patch_size=4,
        #     kernel_size=3,
        #     norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        # ),
        # trans_cfg=dict(
        #     type="DownsampleV1",
        #     patch_size=2,
        #     kernel_size=3,
        #     norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        # ),
        # # for cls only  #消融实验，k=1*1，s=1，p=0;
            #                                                 # k=3*3，s=3，p=1;
            #                                                 # k=5*5，s=5，p=2;
            #                                                 # k=7*7，s=7，p=3
        stem_cfg=dict(
            type="DownsampleV1",
            patch_size=4,
            kernel_size=3,
            norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        ),
        trans_cfg=dict(
            type="DownsampleV1",
            patch_size=2,
            kernel_size=3,
            norm_cfg=dict(type="LayerNorm2d", eps=1e-6),
        ),
        arch_cfg=dict(
            embed_dims=[96, 192, 320, 384],
            num_heads=[2, 4, 8, 16],
            mlp_ratios=[4, 4, 4, 4],
            depths=[2, 2, 4, 2],
        ),
        paca_cfg=dict(
            # default: onsite stage-wise conv-based clustering
            type="P2CConv2d",
            clusters=[100, 100, 100, 0],  # per stage
            # 0: the first block in a stage, 1: true for all blocks in a stage, i > 1: every i blocks
            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            cluster_norm_cfg=dict(type="LN", eps=1e-6),  # for learned clusters
            cluster_pos_embed=False,
            norm_cfg1=dict(type="LN", eps=1e-6),
            norm_cfg2=dict(type="LN", eps=1e-6),
            mlp_func="Mlp",
            with_dwconv_in_mlp=True,
            with_shortcut_in_mlp=True,
        ),
        paca_teacher_cfg=None,  # or
        # paca_teacher_cfg=dict(
        #     type="PaCaTeacher_ConvMixer",
        #     stem_cfg=None,
        #     embed_dims=[96, 192, 320, 384],
        #     depths=[2, 2, 2, 2],
        #     kernel_size=7,
        #     norm_cfg=dict(type="BN"),
        #     act_cfg=dict(type="GELU"),
        #     drop_path_rate=0.0,
        #     return_outs=True,
        # ),
        drop_path_rate=0.0,
        attn_drop=0.0,
        drop=0.0,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        layer_scale=None,
        post_norm=False,
        sub_ln=False,
        with_pos_embed=False,
        out_indices=[],
        downstream_cluster_num=None,
        pretrained=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = arch_cfg["depths"]
        self.num_stages = len(self.depths)
        self.out_indices = out_indices

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))  # 一系列随机衰减的drop概率
        ]  # stochastic depth decay rule
        cur = 0

        paca_cfg_copy = paca_cfg.copy()
        if downstream_cluster_num is not None:         # 根据是否提供downstream_cluster_num，去动态调整clusters
            assert len(downstream_cluster_num) == len(paca_cfg_copy["clusters"])
            paca_cfg_copy["clusters"] = downstream_cluster_num
        clusters = paca_cfg_copy["clusters"]
        onsite_clustering = paca_cfg_copy["onsite_clustering"]
        clustering_kernel_sizes = paca_cfg_copy["clustering_kernel_size"]

        embed_dims = arch_cfg["embed_dims"]
        num_heads = arch_cfg["num_heads"]
        mlp_ratios = arch_cfg["mlp_ratios"]

        # paca teacher 部分：if paca teacher will be used
        self.paca_teacher = None
        if paca_teacher_cfg is not None:
            teacher = paca_teacher_cfg.pop("type")

            return_outs = (
                paca_teacher_cfg["return_outs"]
                and paca_teacher_cfg["embed_dims"] == embed_dims
            )
            paca_teacher_cfg.update(
                dict(
                    in_chans=in_chans,
                    clusters=clusters,
                    return_outs=return_outs,
                )
            )
            self.paca_teacher = eval(teacher)(**paca_teacher_cfg)
            self.share_stem = paca_teacher_cfg["stem_cfg"] is None

        # stem
        stem_cfg_ = stem_cfg.copy()
        stem_cfg_.update(
            dict(
                in_channels=in_chans,
                out_channels=embed_dims[0],
                img_size=img_size,
            )
        )
        self.patch_embed = build_downsample_layer(stem_cfg_)   # 创建 模型的 嵌入层，对input image 下采样
        self.patch_grid = self.patch_embed.grid_size

        # stages，为每个阶段创建对应数量的 PaCaBlock 模块
        for i in range(self.num_stages):
            paca_cfg_ = paca_cfg_copy.copy()
            paca_cfg_["clusters"] = clusters[i]
            paca_cfg_["clustering_kernel_size"] = clustering_kernel_sizes[i]

            blocks = nn.ModuleList()
            # 遍历当前阶段的每个子层
            for j in range(self.depths[i]):
                paca_cfg_cur = paca_cfg_.copy()
                if self.paca_teacher is not None and clusters[i] > 0:
                    paca_cfg_cur["onsite_clustering"] = False       # 使用 PaCaTeacher 且当前阶段需要进行聚类
                else:
                    if j == 0:                                      # 当前阶段的第一个块 进行onsite cluster
                        paca_cfg_cur["onsite_clustering"] = True
                    else:
                        if onsite_clustering[i] < 2:
                            paca_cfg_cur["onsite_clustering"] = onsite_clustering[i]  # 按照给定的onsite_clustering[i]进行onsite cluster
                        else:
                            paca_cfg_cur["onsite_clustering"] = (
                                True if j % onsite_clustering[i] == 0 else False  # 不用PaCaTeacher，每隔 onsite_clustering[i] 个块进行onsite cluster
                            )

                # 创建一个PaCaBlock 模块，并添加到blocks列表
                blocks.append(
                    PaCaBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        paca_cfg=paca_cfg_cur,
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        attn_drop=attn_drop,
                        drop=drop,
                        act_cfg=act_cfg,
                        layer_scale=layer_scale,
                        input_resolution=(
                            self.patch_grid[0] // (2**i),
                            self.patch_grid[1] // (2**i),
                        ),
                        with_pos_embed=with_pos_embed if j == 0 else False,
                        post_norm=post_norm,
                        sub_ln=sub_ln,
                    )
                )
            cur += self.depths[i]  # 跟踪当前已经创建的 PaCaBlock 数量

            setattr(self, f"stage{i + 1}", blocks)    # 将创建的 PaCaBlock 添加到模型中，通过 setattr 方法设置为模型的属性

            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            setattr(self, f"norm{i + 1}", norm)       # 将归一化层设置为模型的属性

            # 创建阶段间的过渡层
            if i < self.num_stages - 1:
                cfg_ = trans_cfg.copy()                         # 复制一份过渡层的配置
                cfg_.update(
                    dict(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[i + 1],
                        img_size=(
                            self.patch_grid[0] // (2**i),
                            self.patch_grid[1] // (2**i),
                        ),
                    )
                )
                transition = build_downsample_layer(cfg_)        # 构建了一个过渡层
                setattr(self, f"transition{i + 1}", transition)  # 通过 setattr 将 过渡层 设置为模型的属性

        # classification 部分
        self.head = None
        if num_classes > 0:
            self.head = nn.Linear(embed_dims[-1], num_classes)

        self.init_weights()
        self.load_pretrained_chkpt(pretrained)

    # 权重初始化：不同类型的层采用不同的权重初始化策略
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):                     # Linear，对权重进行截断正态分布初始化，标准差为 0.02
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)): # LayerNorm、BatchNorm2d（归一化层）
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                              # 存在bias，置为0
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)                          # 存在weight，置为1.0
            elif isinstance(m, nn.Conv2d):                      # Conv2d层
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # fan_out公式：是权重矩阵的输出通道数和卷积核形状的函数
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))             # 用正态分布进行初始化
                if m.bias is not None:
                    m.bias.data.zero_()                                        # 存在bias，置为0

    def load_pretrained_chkpt(self, pretrained=None):
        if isinstance(pretrained, str) and os.path.exists(pretrained):
            load_checkpoint(
                self, pretrained, map_location="cpu", strict=False, logger=None
            )

    def forward_features(self, x):
        # x: B C H W
        x_ = x

        x = self.patch_embed(x)

        # paca teacher
        cluster_assignment = None
        teacher_outs = None
        if self.paca_teacher is not None:
            if self.share_stem:
                x_ = x
            cluster_assignment, teacher_outs = self.paca_teacher(x_)

        # 多阶段特征提取
        outs = []
        HWs = []
        # 遍历每个阶段
        for i in range(self.num_stages):
            H, W = x.shape[2:]
            x = rearrange(x, "B C H W -> B (H W) C").contiguous()
            blocks = getattr(self, f"stage{i + 1}")
            z = None

            if cluster_assignment is not None and i < len(cluster_assignment):  # 处理 PACA Teacher 的聚类分配
                z = cluster_assignment[i]
            # 在每个阶段，通过遍历该阶段的每个 PaCaBlock 模块，逐步提取特征。
            for block in blocks:
                x, z = block(x, H, W, z)

            # 如果有 PACA Teacher，将其输出与 PaCaBlock 的输出相加
            if teacher_outs is not None and i < len(teacher_outs):
                x = x + teacher_outs[i]

            norm = getattr(self, f"norm{i+1}")
            x = norm(x)

            # 处理输出
            if self.head is None and i in self.out_indices:
                outs.append(x)
                HWs.append((H, W))

            # 处理过渡层
            if i != self.num_stages - 1:
                x = c_rearrange(x, H, W)
                transition = getattr(self, f"transition{i + 1}")
                x = transition(x)

        # 处理最终输出：由所有阶段的输出组成
        if self.head is None:
            outs_ = []
            for out, HW in zip(outs, HWs):
                out = c_rearrange(out, HW[0], HW[1])
                outs_.append(out)
            return outs_

        return x

    def forward(self, x):
        x = self.forward_features(x)  # 调用模型的forward_features过程，得到模型的输出 x

        if self.head is not None:  # 如果模型没有头部（head为None），则在每个输出阶段记录输出及其对应的高度和宽度
                                   # 如果有头部，则对所有阶段的输出进行平均池化，然后传递给头部进行最终的分类
            x = x.mean(dim=1)
            x = self.head(x)
            return x

        return x

    def forward_dummy(self, x):         # 用于验证网络结构，不执行实际的训练过程
        x = self.forward_features(x)

        if self.head is not None:
            x = x.mean(dim=1)
            x = self.head(x)
            return x

        return x

_arch_settings = dict(
    tiny=dict(
        embed_dims=[96, 192, 320, 384],
        num_heads=[2, 4, 8, 16],
        # num_heads=[3, 6, 10, 12], # to leverage efficient MHSA in xformers
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 4, 2],
        layer_scale=None,
        drop_path_rate=0.1,
    ),
    tiny_teacher=dict(
        embed_dims=[96, 192, 320, 384],
        num_heads=[2, 4, 8, 16],
        # num_heads=[3, 6, 10, 12], # to leverage efficient MHSA in xformers
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 2, 2],
        teacher_depths=[2, 2, 2, 2],
        layer_scale=None,
        drop_path_rate=0.1,
    ))

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 101,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


model_urls = {
    "paca_placeholder": "ccsn_tiny_p2cconv_100_0_impro"
}

@register_model
def ccsn_tiny_p2cconv_100_0_impro(pretrained=False, **kwargs):
    arch_cfg = _arch_settings["tiny"]
    drop_path_rate = arch_cfg.pop("drop_path_rate", 0.1)
    layer_scale = arch_cfg.pop("layer_scale", None)

    args = dict(
        num_classes=kwargs.pop("num_classes", 101),
        img_size=kwargs.pop("image_size", 224),
        arch_cfg=arch_cfg,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )

    model = CCSN(**args)

    model.pretrained_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(   # 加载预训练权重的检查点
            url=model_urls["ccsn_tiny_p2cconv_100_0_impro"], map_location="cpu"
        )
        state_dict = checkpoint['state_dict']
        # print(state_dict)
        model.load_state_dict(state_dict, strict=False)                  # 加载的权重应用到模型上

    return model

@register_model
def ccsn_tiny_som_100_0(pretrained=False, **kwargs):
    arch_cfg = _arch_settings["tiny"]
    drop_path_rate = arch_cfg.pop("drop_path_rate", 0.1)
    layer_scale = arch_cfg.pop("layer_scale", None)

    args = dict(
        num_classes=kwargs.pop("num_classes", 101),
        img_size=kwargs.pop("image_size", 224),
        arch_cfg=arch_cfg,
        paca_cfg=dict(
            type="CyrusSOM",
            clusters=[100, 100, 100, 0],  # per stage
            # 0: the first block in a stage, 1: true for all blocks in a stage, i > 1: every i blocks
            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            cluster_norm_cfg=dict(type="LN", eps=1e-6),  # for learned clusters
            cluster_pos_embed=False,
            norm_cfg1=dict(type="LN", eps=1e-6),
            norm_cfg2=dict(type="LN", eps=1e-6),
            mlp_func="Mlp",
            with_dwconv_in_mlp=True,
            with_shortcut_in_mlp=True,
        ),
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )

    model = CCSN(**args)

    model.pretrained_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=model_urls["ccsn_tiny_som_100_0"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)

    return model

# --- external clustering teacher, stage-wise
@register_model
def ccsn_convmixer_tiny_100(pretrained=False, **kwargs):
    arch_cfg = _arch_settings["tiny_teacher"]
    drop_path_rate = arch_cfg.pop("drop_path_rate", 0.1)
    layer_scale = arch_cfg.pop("layer_scale", None)
    teacher_depths = arch_cfg.pop("teacher_depths")

    args = dict(
        num_classes=kwargs.pop("num_classes", 101),
        img_size=kwargs.pop("image_size", 224),
        arch_cfg=arch_cfg,
        paca_cfg=dict(
            type="P2CConv2d",
            clusters=[100, 100, 100, 100],  # per stage
            # 0: the first block in a stage, 1: true for all blocks in a stage, i > 1: every i blocks
            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            recurrent_clustering=False,
            cluster_norm_cfg=dict(type="LN", eps=1e-6),  # for learned clusters
            cluster_pos_embed=False,
            norm_cfg1=dict(type="LN", eps=1e-6),
            norm_cfg2=dict(type="LN", eps=1e-6),
            mlp_func="Mlp",
            with_dwconv_in_mlp=True,
            with_shortcut_in_mlp=True,
        ),
        paca_teacher_cfg=dict(
            type="PaCaTeacher_ConvMixer",
            stem_cfg=None,
            embed_dims=arch_cfg["embed_dims"],
            depths=teacher_depths,
            kernel_size=7,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="GELU"),
            drop_path_rate=0.0,
            return_outs=True,
        ),
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )

    model = CCSN(**args)

    model.pretrained_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=model_urls["ccsn_convmixer_tiny_100"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)

    return model

### test ---------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# run it in standalone by: python3 -m models.CCSN
if __name__ == "__main__":
    img, num_classes = torch.randn(64, 3, 224, 224), 101
    img = img.to(memory_format=torch.channels_last)

    models = [
        "ccsn_tiny_p2cconv_100_0_impro",
        # "ccsn_convmixer_tiny_100",
        "ccsn_tiny_som_100_0",
              ]

    for i, model_name in enumerate(models):
        model = eval(model_name)(num_classes=num_classes)          # 实例化模型
        model = model.to(memory_format=torch.channels_last)
        out = model(img)                                           # 在输入图像 img 上运行模型
        if i == 0:
            print(model)
        print(f"{model_name}:", out.shape, count_parameters(model)) # 打印模型结构、名称、输出形状、参数量
