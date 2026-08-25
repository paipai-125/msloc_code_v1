from transformers import XCLIPVisionModel
import json
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .clip import clip
import math

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .clip import pscan

@dataclass
class MambaConfig:
    d_model: 768 # D
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    drop_prob: float = 0.1

    bias: bool = False
    conv_bias: bool = True
    bimamba: bool = True

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.drop_path = DropPath(drop_prob=config.drop_prob)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.drop_path(self.mixer(self.norm(x))) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        self.bimamba = config.bimamba

        if self.bimamba:
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))

            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(config.d_inner))

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        if self.bimamba:
            x_b = x.flip([-1])
            A_b = -torch.exp(self.A_b_log.float()) # (ED, N)
            D_b = self.D_b.float()
            deltaBC_b = self.x_proj_b(x_b)
            delta_b, B_b, C_b = torch.split(deltaBC_b, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
            delta_b = F.softplus(self.dt_proj_b(delta_b)) # (B, L, ED)
            if self.config.pscan:
                y_b = self.selective_scan(x_b, delta_b, A_b, B_b, C_b, D_b)
            else:
                y_b = self.selective_scan_seq(x_b, delta_b, A_b, B_b, C_b, D_b)
            y_b = y_b.flip([-1])
            y = y + y_b
        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        # todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data

class XCLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=768, class_num=1, xclip_model_path=None
    ):
        super(XCLIP_DeMamba, self).__init__()
        # XCLIP checkpoint path is supplied by config; the sibling asset path is a legacy fallback.
        _xclip_path = xclip_model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "MSLoc_data", "DeMamba", "pretrained_weights", "xclip-base-patch16",
        )
        # This project always uses a checkpoint directory supplied locally.
        # ``local_files_only`` makes a missing/incomplete directory fail fast
        # instead of silently attempting a Hugging Face download.
        self.encoder = XCLIPVisionModel.from_pretrained(_xclip_path, local_files_only=True)
        blocks = []
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear((self.patch_nums+1)*channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.initialize_weights(self.fc1)
        self.dropout = nn.Dropout(p=0.0)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        sequence_output = sequence_output.view(b, t, -1, c)
        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)

        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)
        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(video_level_features)
        pred = self.dropout(pred)

        return pred


class XCLIP_NeuronDeMamba(nn.Module):
    """Frozen XCLIP + 768 selected neurons + trainable Mamba classifier.

    The selector file is produced by ``probe_xclip_neurons.py``.  It records
    channels from the *encoder* hidden states, numbered from one (the embedding
    output at ``hidden_states[0]`` is intentionally never selected).  This is
    tailored to ``microsoft/xclip-base-patch16``: a 224px, patch-16 ViT whose
    vision width is 768.  Patch and layer counts are still read from the loaded
    checkpoint instead of being hard-coded.
    """

    def __init__(self, neuron_indices_path, xclip_model_path, class_num=4):
        super().__init__()
        if not xclip_model_path:
            raise ValueError("XCLIP_NeuronDeMamba requires a local xclip_model_path")
        xclip_path = os.path.abspath(xclip_model_path)
        if not os.path.isdir(xclip_path):
            raise FileNotFoundError(f"Local XCLIP model directory not found: {xclip_path}")
        # Do not fall back to Hugging Face Hub: the checkpoint is local.
        self.encoder = XCLIPVisionModel.from_pretrained(xclip_path, local_files_only=True)

        config = self.encoder.config
        self.hidden_size = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.image_size = int(config.image_size)
        self.patch_nums = (self.image_size // self.patch_size) ** 2
        if self.hidden_size != 768:
            raise ValueError(
                "XCLIP_NeuronDeMamba expects microsoft/xclip-base-patch16 "
                f"(hidden_size=768), but loaded hidden_size={self.hidden_size}."
            )

        selector_path = os.path.abspath(neuron_indices_path)
        if not os.path.isfile(selector_path):
            raise FileNotFoundError(f"Neuron selector file not found: {selector_path}")
        with open(selector_path, "r", encoding="utf-8") as handle:
            selector = json.load(handle)
        layers = selector.get("layers", selector.get("selected_indices", selector))
        if not isinstance(layers, dict):
            raise ValueError(f"Invalid neuron selector file: {selector_path}")

        self.selected_layer_numbers = []
        selected_width = 0
        for raw_layer, raw_indices in sorted(layers.items(), key=lambda item: self._selector_layer_number(item[0])):
            layer_number = self._selector_layer_number(raw_layer)
            indices = torch.as_tensor(raw_indices, dtype=torch.long)
            if layer_number < 1 or indices.ndim != 1 or indices.numel() == 0:
                raise ValueError(f"Invalid selector for layer {raw_layer!r}")
            if int(indices.min()) < 0 or int(indices.max()) >= self.hidden_size:
                raise ValueError(f"Selector indices for layer {raw_layer!r} are outside [0, {self.hidden_size})")
            if indices.unique().numel() != indices.numel():
                raise ValueError(f"Selector indices for layer {raw_layer!r} contain duplicates")
            buffer_name = f"neuron_indices_l{layer_number}"
            self.register_buffer(buffer_name, indices, persistent=True)
            self.selected_layer_numbers.append(layer_number)
            selected_width += int(indices.numel())
        if not self.selected_layer_numbers:
            raise ValueError("Neuron selector contains no selected XCLIP layers")

        self.selected_width = selected_width
        if self.selected_width != self.hidden_size:
            raise ValueError(
                "The direct-input neuron model requires exactly 768 selected neurons "
                f"to preserve the original Mamba width, but selector contains {self.selected_width}. "
                "Re-run probe_xclip_neurons.py with --final-neuron-count 768."
            )
        # There is deliberately no projection/adapter here.  The probe emits
        # exactly 768 channels, which directly retain the original Mamba width.
        self.mamba_configs = MambaConfig(d_model=self.hidden_size)
        self.mamba = ResidualBlock(config=self.mamba_configs)
        self.fc_norm = nn.LayerNorm(self.patch_nums * self.hidden_size)
        self.fc_norm2 = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear((self.patch_nums + 1) * self.hidden_size, class_num)
        self.dropout = nn.Dropout(p=0.0)
        self.initialize_weights(self.fc1)

    @staticmethod
    def _selector_layer_number(name):
        text = str(name)
        if text.startswith("layer_"):
            text = text[len("layer_"):]
        return int(text)

    @staticmethod
    def initialize_weights(module):
        for item in module.modules():
            if isinstance(item, nn.Linear):
                init.xavier_uniform_(item.weight)
                if item.bias is not None:
                    init.constant_(item.bias, 0)

    def train(self, mode=True):
        """Keep the frozen XCLIP deterministic when the outer model trains."""
        super().train(mode)
        self.encoder.eval()
        return self

    def _selected_patch_features(self, hidden_states, batch_size, num_frames):
        selected = []
        for layer_number in self.selected_layer_numbers:
            if layer_number >= len(hidden_states):
                raise ValueError(
                    f"Selector requests XCLIP layer {layer_number}, but the loaded "
                    f"model returned {len(hidden_states) - 1} encoder hidden states."
                )
            hidden = hidden_states[layer_number]
            patch_features = hidden[:, 1:, :]
            if patch_features.shape[1] != self.patch_nums:
                raise ValueError(
                    f"Expected {self.patch_nums} patch tokens from XCLIP, got {patch_features.shape[1]}. "
                    "Use the matching xclip-base-patch16 processor and image size."
                )
            patch_features = patch_features.reshape(batch_size, num_frames, self.patch_nums, self.hidden_size)
            indices = getattr(self, f"neuron_indices_l{layer_number}")
            selected.append(patch_features.index_select(-1, indices))
        return torch.cat(selected, dim=-1)

    def forward(self, x):
        batch_size, num_frames, _, height, width = x.shape
        images = x.reshape(batch_size * num_frames, 3, height, width)
        outputs = self.encoder(images, output_hidden_states=True)
        selected = self._selected_patch_features(outputs.hidden_states, batch_size, num_frames)

        features = selected.to(dtype=self.fc1.weight.dtype)
        global_feat = self.fc_norm2(features.mean(dim=(1, 2)))

        grid_size = int(math.isqrt(self.patch_nums))
        if grid_size * grid_size != self.patch_nums:
            raise ValueError(f"XCLIP patch token count must form a square grid, got {self.patch_nums}")
        # Preserve the original DeMamba organisation: one temporal sequence per
        # spatial patch, then a bidirectional state-space mixer over frames.
        mamba_input = features.reshape(batch_size, num_frames, 1, grid_size, 1, grid_size, self.hidden_size)
        mamba_input = mamba_input.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        mamba_input = mamba_input.reshape(batch_size * self.patch_nums, num_frames, self.hidden_size)
        mamba_output = self.mamba(mamba_input)

        local_feat = mamba_output.mean(dim=1).reshape(batch_size, -1)
        local_feat = self.fc_norm(local_feat)
        logits = self.fc1(torch.cat((global_feat, local_feat), dim=1))
        return self.dropout(logits)



class CLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=512, class_num=1
    ):
        super(CLIP_DeMamba, self).__init__()
        self.clip_model, preprocess = clip.load('ViT-B-14')
        self.clip_model = self.clip_model.float()
        blocks = []
        channel = 512
        self.fusing_ratios = 2
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear(channel*(self.patch_nums+1), class_num)
        self.bn1 = nn.BatchNorm1d(channel)
        self.initialize_weights(self.fc1)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, -1, c)

        global_feat = sequence_output.reshape(b, -1, c)
        global_feat = global_feat.mean(1)

        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)
        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)

        video_level_features = torch.cat((global_feat, video_level_features), dim=1)
        x = self.fc1(video_level_features)

        return x

if __name__ == '__main__':
    model = CLIP_DeMamba()
    print(model)
