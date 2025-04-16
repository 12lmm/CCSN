
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

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, LayerNorm2d
from timm.layers.norm import _is_contiguous
from timm.models._registry import register_model


from mmengine.registry import MODELS
from mmengine.model import constant_init, kaiming_init
from mmengine.runner import load_checkpoint

from mmcv.cnn import (
    build_norm_layer,
    build_activation_layer,
)

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


try:
    import xformers.ops as xops

    has_xformers = True
except ImportError:
    has_xformers = False


from .layers.blur_pool import BlurConv2d
from .layers.downsample import build_downsample_layer


__all__ = ["CCSN"]


def c_rearrange(x, H, W, dim=1):
    channels_last = x.is_contiguous(memory_format=torch.channels_last)
    if dim == 1:
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
    elif dim == 2:
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
    else:
        raise NotImplementedError

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        x = x.contiguous()
    return x


class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=True, with_shortcut=False):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, 1, (kernel_size - 1) // 2, bias=bias, groups=dim
        )
        self.with_shortcut = with_shortcut

    def forward(self, x, H, W):
        shortcut = x
        x = c_rearrange(x, H, W)
        x = self.dwconv(x)
        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W)
        if self.with_shortcut:
            return x + shortcut
        return x


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
        self.dwconv = (
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


class Residual(nn.Module):
    def __init__(self, fn, drop_path_rate=0.0):
        super().__init__()
        self.fn = fn

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        return self.drop_path(self.fn(x)) + x



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

        self.clustering = nn.Sequential(
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
        label = []
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[0])
        class_dict = {}
        for index in range(self.train_x.shape[0]):
            if label[index] in class_dict.keys():
                class_dict[label[index]].append(index)
            else:
                class_dict[label[index]] = [index]


        B = x.shape[0]
        M = 100
        H = x.shape[2]
        W = x.shape[3]

        d = x[:, :, 0, 0]
        x = d
        x = np.array(x.detach().cpu())

        cluster_center = {}
        for key, value in class_dict.items():
            cluster_center[key] = np.array([x[i, :] for i in value]).mean(axis=0)
        self.cluster_center = cluster_center
        # print(f"cluster_center: {cluster_center}")

        clustered_features = torch.zeros(B, M, H, W)



        for key, value in cluster_center.items():

            value_tensor = torch.tensor(value).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            m = value_tensor.shape[1]
            value = value_tensor.expand(B, m, H, W)

            clustered_features[:, key, :, :] = value[:, key, :, :]
        clustered_features = clustered_features.view(B, M, H * W)



        return clustered_features

    def __call__(self, x, H, W):
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()

        return self.forward(x)


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
        if self.num_clusters > 0:
            self.cluster_norm = (
                build_norm_layer(paca_cfg["cluster_norm_cfg"], dim)[1]
                if paca_cfg["cluster_norm_cfg"]
                else nn.Identity()
            )

            if paca_cfg["type"] == "CyrusSOM":
                self.clustering = eval(paca_cfg["type"])(

                )
            else:
                self.clustering = eval(paca_cfg["type"])(
                    dim=dim,
                    num_clusters=self.num_clusters,
                    mlp_ratio=mlp_ratio,
                    kernel_size=paca_cfg["clustering_kernel_size"],
                    act_cfg=act_cfg,
                )


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

        self.attn_viz = nn.Identity()

    def forward(self, x, H, W, z, alpha1=0.2, weight_factor=2.0):

        if self.num_clusters > 0:
            if self.onsite_clustering:
                z_raw = self.clustering(x, H, W)
                z = z_raw.softmax(dim=-1)



            top2_conf, top2_indices = torch.topk(z, k=2, dim=-1)
            top1_conf = top2_conf[:, :, 0]
            top2_conf_val = top2_conf[:, :, 1]


            conf_diff = top1_conf - top2_conf_val

            #
            alpha2 = alpha1 * 0.5


            adaptive_factor = torch.mean(conf_diff) * 0.5
            weight_factor = adaptive_factor + weight_factor

            reliable_samples = (top1_conf >= alpha1) & (conf_diff >= alpha2)


            weight = reliable_samples.float() * weight_factor + (1 - reliable_samples.float())

            # c = z @ x  # B M C
            c2 = einsum("bmn,bnc->bmc", z, x)

            c_weighted = c2 * weight.unsqueeze(-1)

            if self.cluster_pos_embed:
                c_weighted = c_weighted  + self.cluster_pos_enc.expand(c_weighted.shape[0], -1, -1)
            c = self.cluster_norm(c_weighted)

        else:
            c = x
            weight = torch.ones_like(x)



        if self.use_xformers:

            q = self.q(x)
            k = self.k(c)
            v = self.v(c)
            q = rearrange(q, "B N (h d) -> B N h d", h=self.num_heads)
            k = rearrange(k, "B M (h d) -> B M h d", h=self.num_heads)
            v = rearrange(v, "B M (h d) -> B M h d", h=self.num_heads)

            x = xops.memory_efficient_attention(q, k, v)
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
                need_weights=not self.training,
                average_attn_weights=False,
            )

            x = rearrange(x, "N B C -> B N C")

            if not self.training:
                attn = self.attn_viz(attn)

        x = self.proj_drop(x)

        return x, z



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
        sub_ln=False,
        **kwargs,
    ):
        super().__init__()

        self.post_norm = post_norm

        self.with_pos_embed = with_pos_embed
        self.input_resolution = input_resolution
        if self.with_pos_embed:
            assert self.input_resolution is not None
            self.input_resolution = to_2tuple(self.input_resolution)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.input_resolution[0] * self.input_resolution[1], dim)
            )
            self.pos_drop = nn.Dropout(p=drop)
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm1_before = (
            build_norm_layer(paca_cfg["norm_cfg1"], dim)[1]
            if sub_ln or not post_norm
            else nn.Identity()
        )
        self.attn = PaCaLayer(
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

        self.mlp = eval(paca_cfg["mlp_func"])(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            with_dwconv=paca_cfg["with_dwconv_in_mlp"],
            with_shortcut=paca_cfg["with_shortcut_in_mlp"],
            act_cfg=act_cfg,
        )
        self.norm2_after = (
            build_norm_layer(paca_cfg["norm_cfg2"], dim)[1]
            if sub_ln or post_norm
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x, H, W, z):
        # x: B N C
        if self.with_pos_embed:
            if self.input_resolution != (H, W):
                pos_embed = rearrange(self.pos_embed, "B (H W) C -> B C H W")
                pos_embed = F.interpolate(
                    pos_embed, size=(H, W), mode="bilinear", align_corners=True
                )
                pos_embed = rearrange(pos_embed, "B C H W -> B (H W) C")
            else:
                pos_embed = self.pos_embed

            x = self.pos_drop(x + pos_embed)

        a, z = self.attn(self.norm1_before(x), H, W, z)
        a = self.norm1_after(a)
        if not self.layer_scale:
            x = x + self.drop_path1(a)
            x = x + self.drop_path2(
                self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )
        else:
            x = x + self.drop_path1(self.gamma1 * a)
            x = x + self.drop_path2(
                self.gamma2 * self.norm2_after(self.mlp(self.norm2_before(x), H, W))
            )

        return x, z


class CCSN(nn.Module):


    def __init__(
        self,
        in_chans=3,
        num_classes=101,
        img_size=224,

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

            type="P2CConv2d",
            clusters=[100, 100, 100, 0],
            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            cluster_norm_cfg=dict(type="LN", eps=1e-6),
            cluster_pos_embed=False,
            norm_cfg1=dict(type="LN", eps=1e-6),
            norm_cfg2=dict(type="LN", eps=1e-6),
            mlp_func="Mlp",
            with_dwconv_in_mlp=True,
            with_shortcut_in_mlp=True,
        ),


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
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule
        cur = 0

        paca_cfg_copy = paca_cfg.copy()
        if downstream_cluster_num is not None:
            assert len(downstream_cluster_num) == len(paca_cfg_copy["clusters"])
            paca_cfg_copy["clusters"] = downstream_cluster_num
        clusters = paca_cfg_copy["clusters"]
        onsite_clustering = paca_cfg_copy["onsite_clustering"]
        clustering_kernel_sizes = paca_cfg_copy["clustering_kernel_size"]

        embed_dims = arch_cfg["embed_dims"]
        num_heads = arch_cfg["num_heads"]
        mlp_ratios = arch_cfg["mlp_ratios"]


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
        self.patch_embed = build_downsample_layer(stem_cfg_)
        self.patch_grid = self.patch_embed.grid_size


        for i in range(self.num_stages):
            paca_cfg_ = paca_cfg_copy.copy()
            paca_cfg_["clusters"] = clusters[i]
            paca_cfg_["clustering_kernel_size"] = clustering_kernel_sizes[i]

            blocks = nn.ModuleList()

            for j in range(self.depths[i]):
                paca_cfg_cur = paca_cfg_.copy()
                if self.paca_teacher is not None and clusters[i] > 0:
                    paca_cfg_cur["onsite_clustering"] = False
                else:
                    if j == 0:
                        paca_cfg_cur["onsite_clustering"] = True
                    else:
                        if onsite_clustering[i] < 2:
                            paca_cfg_cur["onsite_clustering"] = onsite_clustering[i]
                        else:
                            paca_cfg_cur["onsite_clustering"] = (
                                True if j % onsite_clustering[i] == 0 else False
                            )


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
            cur += self.depths[i]

            setattr(self, f"stage{i + 1}", blocks)

            norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            setattr(self, f"norm{i + 1}", norm)


            if i < self.num_stages - 1:
                cfg_ = trans_cfg.copy()
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
                transition = build_downsample_layer(cfg_)
                setattr(self, f"transition{i + 1}", transition)


        self.head = None
        if num_classes > 0:
            self.head = nn.Linear(embed_dims[-1], num_classes)

        self.init_weights()
        self.load_pretrained_chkpt(pretrained)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

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


        outs = []
        HWs = []

        for i in range(self.num_stages):
            H, W = x.shape[2:]
            x = rearrange(x, "B C H W -> B (H W) C").contiguous()
            blocks = getattr(self, f"stage{i + 1}")
            z = None

            if cluster_assignment is not None and i < len(cluster_assignment):
                z = cluster_assignment[i]

            for block in blocks:
                x, z = block(x, H, W, z)


            if teacher_outs is not None and i < len(teacher_outs):
                x = x + teacher_outs[i]

            norm = getattr(self, f"norm{i+1}")
            x = norm(x)


            if self.head is None and i in self.out_indices:
                outs.append(x)
                HWs.append((H, W))


            if i != self.num_stages - 1:
                x = c_rearrange(x, H, W)
                transition = getattr(self, f"transition{i + 1}")
                x = transition(x)


        if self.head is None:
            outs_ = []
            for out, HW in zip(outs, HWs):
                out = c_rearrange(out, HW[0], HW[1])
                outs_.append(out)
            return outs_

        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.head is not None:
            x = x.mean(dim=1)
            x = self.head(x)
            return x

        return x

    def forward_dummy(self, x):
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
        # num_heads=[3, 6, 10, 12],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 4, 2],
        layer_scale=None,
        drop_path_rate=0.1,
    ),)

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
    "paca_placeholder": "ccsn"
}

@register_model
def ccsn(pretrained=False, **kwargs):
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
        checkpoint = torch.hub.load_state_dict_from_url(
            url=model_urls["ccsn"], map_location="cpu"
        )
        state_dict = checkpoint['state_dict']
        # print(state_dict)
        model.load_state_dict(state_dict, strict=False)

    return model

@register_model
def CCSN(pretrained=False, **kwargs):
    arch_cfg = _arch_settings["tiny"]
    drop_path_rate = arch_cfg.pop("drop_path_rate", 0.1)
    layer_scale = arch_cfg.pop("layer_scale", None)

    args = dict(
        num_classes=kwargs.pop("num_classes", 101),
        img_size=kwargs.pop("image_size", 224),
        arch_cfg=arch_cfg,
        paca_cfg=dict(
            type="CyrusSOM",
            clusters=[100, 100, 100, 0],

            onsite_clustering=[0, 0, 0, 0],
            clustering_kernel_size=[7, 7, 7, 7],
            cluster_norm_cfg=dict(type="LN", eps=1e-6),
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
            url=model_urls["ccsn"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)

    return model

    model = CCSN(**args)

    model.pretrained_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=model_urls["ccsn"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)

    return model

### test ---------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6



if __name__ == "__main__":
    img, num_classes = torch.randn(64, 3, 224, 224), 101
    img = img.to(memory_format=torch.channels_last)

    models = ["ccsn"]

    for i, model_name in enumerate(models):
        model = eval(model_name)(num_classes=num_classes)
        model = model.to(memory_format=torch.channels_last)
        out = model(img)
        if i == 0:
            print(model)
        print(f"{model_name}:", out.shape, count_parameters(model))
