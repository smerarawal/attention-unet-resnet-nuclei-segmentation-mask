"""Attention UNet++ generalised to any 5-level encoder, with two heads:
   sem: (B, n_classes, H, W) logits   |   bnd: (B, 1, H, W) boundary logit (for instance splitting)
Node (i, j): x[i][j] = DecoderBlock(cat[ AG(g=up(x[i+1][j-1]), skips=cat(x[i][0..j-1])), g ])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionGate, DecoderBlock, conv_bn_relu
from .encoders import build_encoder


class AttnUNetPP(nn.Module):
    def __init__(self, encoder, n_classes=5, base_ch=32):
        super().__init__()
        self.encoder = encoder
        enc = encoder.out_channels
        dec = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.gates, self.nodes = nn.ModuleDict(), nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                g_ch = enc[i + 1] if j == 1 else dec[i + 1]
                skip_ch = enc[i] + dec[i] * (j - 1)
                self.gates[f"{i}_{j}"] = AttentionGate(g_ch, skip_ch, max(dec[i] // 2, 8))
                self.nodes[f"{i}_{j}"] = DecoderBlock(skip_ch + g_ch, dec[i])
        self.refine = conv_bn_relu(dec[0], dec[0]) if encoder.first_stride == 2 else nn.Identity()
        self.sem_head = nn.Conv2d(dec[0], n_classes, 1)
        self.bnd_head = nn.Conv2d(dec[0], 1, 1)

    def set_keep_attention(self, flag: bool):
        for g in self.gates.values():
            g.keep = flag

    def attention_maps(self):
        """{'i_j': (B,1,h,w) psi} from the last forward with keep=True."""
        return {k: g.last_psi for k, g in self.gates.items() if g.last_psi is not None}

    def forward(self, x):
        H, W = x.shape[-2:]
        X = {(i, 0): f for i, f in enumerate(self.encoder(x))}
        for j in range(1, 5):
            for i in range(0, 5 - j):
                skip = torch.cat([X[(i, k)] for k in range(j)], 1)
                g = F.interpolate(X[(i + 1, j - 1)], size=skip.shape[-2:], mode="bilinear", align_corners=False)
                att = self.gates[f"{i}_{j}"](g, skip)
                X[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat([att, g], 1))
        out = X[(0, 4)]
        if self.encoder.first_stride == 2:
            out = self.refine(F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False))
        return self.sem_head(out), self.bnd_head(out)


def build_model(cfg):
    enc = build_encoder(cfg.encoder, cfg.pretrained, cfg.encoder_weights_path, cfg.base_ch)
    return AttnUNetPP(enc, cfg.n_classes, cfg.base_ch)
