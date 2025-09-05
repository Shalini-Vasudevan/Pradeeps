import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Windows-friendly imports
try:
    from csms6s import (
        CrossScan, CrossMerge, CrossScan_fs_ft, CrossScan_fs_bt,
        CrossScan_bs_ft, CrossScan_bs_bt, CrossMerge_bs_bt, CrossMerge_bs_ft,
        CrossMerge_fs_bt, CrossMerge_fs_ft, CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs,
        CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction,
        SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex,
        flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
    )
    from csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F, getCSM
except ImportError:
    # fallback if running in different folder structure
    from .csms6s import (
        CrossScan, CrossMerge, CrossScan_fs_ft, CrossScan_fs_bt,
        CrossScan_bs_ft, CrossScan_bs_bt, CrossMerge_bs_bt, CrossMerge_bs_ft,
        CrossMerge_fs_bt, CrossMerge_fs_ft, CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs,
        CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction,
        SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex,
        flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
    )
    from .csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F, getCSM


# ===== Helper Classes =====
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ===== BiSTSSM Wrapper =====
class BiSTSSM(nn.Module):
    def __init__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", act_layer=nn.SiLU,
                 d_conv=3, conv_bias=True, dropout=0.0, bias=False, dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, initialize="v0",
                 forward_type="v2", channel_first=False):
        super().__init__()
        # Initialize your full BiSTSSM_v2 here with all necessary arguments
        from types import SimpleNamespace
        self.model = SimpleNamespace()  # placeholder for BiSTSSM_v2 init on Windows

    def forward(self, x: torch.Tensor):
        return x  # placeholder forward


class BiSTSSMBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0., norm_layer=nn.LayerNorm, channel_first=False,
                 ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,
                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0., ssm_init="v0",
                 forward_type="v2", mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.,
                 gmlp=False, use_checkpoint=False, post_norm=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM(d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio,
                              dt_rank=ssm_dt_rank, act_layer=ssm_act_layer, d_conv=ssm_conv,
                              conv_bias=ssm_conv_bias, dropout=ssm_drop_rate, initialize=ssm_init,
                              forward_type=forward_type, channel_first=channel_first)
        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim,
                            act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
