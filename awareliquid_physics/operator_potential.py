"""
operator_potential.py — the potential energy V(q | ctx) as a SPECTRAL OPERATOR.

v0.2 (M2). Replaces the pointwise MLP potential of v0.1 with a Fourier neural
operator (FNO, Li et al., ICLR 2021) so the energy landscape carries real
spatial structure:

  * translation invariance     — the spectral (convolution) layer and the
                                  per-node local layer use the SAME weights at
                                  every node; no coordinate inputs, so
                                  V(q + c) = V(q) for any uniform shift c.
  * resolution invariance      — all parameters live in Fourier space
                                  (truncated modes) or as per-node pointwise
                                  maps, so the SAME module can be evaluated on
                                  any number of nodes N (64 -> 256 zero-shot).
  * smoothness                 — Tanh activations only: the Hamiltonian field
                                  is dV/dq (autograd), so the potential must be
                                  at least C^2 (no ReLU kinks).
  * scalar energy              — the potential is the SUM of per-node pointwise
                                  energies: V(q) = sum_n v(q_n). This keeps
                                  energy extensive and the force local+global.

Non-periodic boundaries are handled by REFLECT padding before each FFT (a
physical choice: the field continues smoothly at the wall, no Gibbs ringing
from a hard zero cut). The spectral weights are stored as a real
(real, imag) stack so standard optimisers (Adam) work unchanged.

Physics contract (enforced by tests):
  V: (B, N, dim) -> (B,)  scalar, translation-invariant, resolution-invariant.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral convolution (FNO layer pieces)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Global convolution on the node axis via FFT: (K v)(x) = IFFT(R . FFT(v)).

    v: (B, C, N) -> (B, C_out, N).  R is a complex (modes, C_in, C_out) tensor
    stored as a real parameter stack [real, imag]; only the `modes` lowest
    frequencies are kept (the smooth physics assumption), which is exactly what
    makes the layer resolution-invariant: no parameter depends on N.
    """

    def __init__(self, modes: int, c_in: int, c_out: int):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.rand(2, self.modes, c_in, c_out))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B, C, N = v.shape
        vf = torch.fft.rfft(v, dim=-1)                       # (B, C, N//2+1)
        n_half = vf.shape[-1]
        # Few nodes -> few available frequencies; truncate adaptively so the
        # layer also works at tiny resolutions (physics: use the modes that exist).
        m = min(self.modes, n_half)
        w = torch.complex(self.weight[0][:m], self.weight[1][:m])  # (m, C_in, C_out)
        vf_m = vf[..., :m]                                   # (B, C, m)
        out = torch.einsum("bcm,mco->bom", vf_m, w)          # (B, C_out, m)
        if out.shape[-1] < n_half:                           # zero the dropped modes
            out = F.pad(out, (0, n_half - out.shape[-1]))
        return torch.fft.irfft(out, n=N, dim=-1)             # (B, C_out, N)


class SpectralConv2d(nn.Module):
    """Global convolution on a 2D node grid via FFT2: (K v)(x,y) =
    IFFT2(R . FFT2(v)). v: (B, C, H, W) -> (B, C_out, H, W). R is a complex
    (modes_x, modes_y, C_in, C_out) tensor stored as a real (real, imag) stack;
    only the low frequencies are kept — the smooth-physics assumption that
    makes the layer resolution-invariant (no parameter depends on H or W)."""

    def __init__(self, modes_x: int, modes_y: int, c_in: int, c_out: int):
        super().__init__()
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(
            scale * torch.rand(2, self.modes_x, self.modes_y, c_in, c_out))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B, C, H, W = v.shape
        vf = torch.fft.rfft2(v, dim=(-2, -1))                # (B, C, H, W//2+1)
        mx = min(self.modes_x, vf.shape[-2])
        my = min(self.modes_y, vf.shape[-1])
        w = torch.complex(self.weight[0][:mx, :my], self.weight[1][:mx, :my])
        vf_m = vf[..., :mx, :my]                              # (B, C, mx, my)
        out = torch.einsum("bchw,hwco->bohw", vf_m, w)        # (B, C_out, mx, my)
        if out.shape[-2:] != vf.shape[-2:]:                   # zero dropped modes
            pad_h = vf.shape[-2] - out.shape[-2]
            pad_w = vf.shape[-1] - out.shape[-1]
            out = F.pad(out, (0, pad_w, 0, pad_h))
        return torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))  # (B, C_out, H, W)


class FNOBlock2d(nn.Module):
    """One 2D FNO layer: v <- Tanh( spectral2d(v) + local(v) ), local = 1x1 conv."""

    def __init__(self, modes_x: int, modes_y: int, width: int):
        super().__init__()
        self.spectral = SpectralConv2d(modes_x, modes_y, width, width)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.spectral(v) + self.local(v))


class FNOBlock(nn.Module):
    """One FNO layer:  v <- Tanh( spectral(v) + local(v) ).

    `local` is a per-node 1x1 conv — the SAME matrix at every node, so it
    preserves translation equivariance while giving each node its own local
    potential contribution (e.g. an external field term)."""

    def __init__(self, modes: int, width: int):
        super().__init__()
        self.spectral = SpectralConv1d(modes, width, width)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.spectral(v) + self.local(v))


class FiLM(nn.Module):
    """Context conditioning: h <- scale(ctx) * h + bias(ctx), per channel.

    Acts on the channel axis only, so it keeps translation equivariance and
    resolution invariance — the context (system identification from the liquid
    core) reshapes the energy landscape without touching its spatial laws."""

    def __init__(self, width: int, context_dim: int):
        super().__init__()
        self.scale = nn.Linear(context_dim, width)
        self.bias = nn.Linear(context_dim, width)
        nn.init.constant_(self.scale.bias, 1.0)   # start as identity modulation
        nn.init.zeros_(self.bias.bias)

    def forward(self, v: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        s = self.scale(context)                  # (B, W)
        b = self.bias(context)
        while s.dim() < v.dim():                 # broadcast to v's trailing dims
            s = s.unsqueeze(-1)                  # (B, W) / (B, W, N) / (B, W, H, W)...
            b = b.unsqueeze(-1)
        return s * v + b


# ---------------------------------------------------------------------------
# The operator potential
# ---------------------------------------------------------------------------

class OperatorPotential(nn.Module):
    """V(q | ctx) = sum_n project(FNO_blocks(lift(q)))_n  — a spectral potential.

    q:      (B, N, dim)    node states; N is ARBITRARY (resolution-invariant)
    context: (B, d_ctx) or None — FiLM conditions every FNO block.
    Returns: (B,) scalar energy.

    Degenerate low-dof case: N=1 (or dim small, single oscillator) makes the
    FFT identity, so the stack reduces to per-node nonlinear maps — behaviour
    compatible with the v0.1 pointwise-MLP potential.
    """

    def __init__(self, dim: int, width: int = 32, modes: int = 12,
                 depth: int = 4, context_dim: int = 0, reflect_pad: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.reflect_pad = int(reflect_pad)
        self.lift = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(
            [FNOBlock(modes, width) for _ in range(depth)])
        self.films = nn.ModuleList(
            [FiLM(width, context_dim) if context_dim > 0 else None
             for _ in range(depth)])
        self.project = nn.Linear(width, 1)

    def forward(self, q: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, dim = q.shape
        assert dim == self.dim, f"q has dim {dim}, potential expects {self.dim}"

        v = self.lift(q)                                   # (B, N, W)
        v = v.permute(0, 2, 1)                             # (B, W, N)

        # Reflect-pad so the non-periodic domain continues smoothly at the
        # boundaries (physical: no Gibbs ringing from a hard zero cut).
        if self.reflect_pad > 0 and N > 2 * self.reflect_pad:
            v = F.pad(v, (self.reflect_pad, self.reflect_pad), mode="reflect")

        for block, film in zip(self.blocks, self.films):
            v = block(v)
            if film is not None:
                assert context is not None, "context_dim > 0 requires a context"
                v = film(v, context)

        if self.reflect_pad > 0 and N > 2 * self.reflect_pad:
            v = v[..., self.reflect_pad: self.reflect_pad + N]
        v = v.permute(0, 2, 1)                             # (B, N, W)

        e = self.project(v).squeeze(-1)                    # (B, N) pointwise energy
        return e.sum(dim=-1)                               # (B,)  extensive scalar


class OperatorPotential2d(nn.Module):
    """V(q | ctx) = sum_{x,y} project(FNO2d_blocks(lift(q)))_{x,y} — the 2D-grid
    version of OperatorPotential (P2 extension: real 2D fields).

    q: (B, H, W, dim) — H, W arbitrary (resolution-invariant: all parameters
    live in Fourier space or as per-node maps). Same physics contract: Tanh
    smoothness, translation invariance, scalar extensive energy, FiLM
    conditioning on the channel axis."""

    def __init__(self, dim: int, width: int = 32, modes_x: int = 12,
                 modes_y: int = 12, depth: int = 4, context_dim: int = 0,
                 reflect_pad: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.reflect_pad = int(reflect_pad)
        self.lift = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(modes_x, modes_y, width) for _ in range(depth)])
        self.films = nn.ModuleList(
            [FiLM(width, context_dim) if context_dim > 0 else None
             for _ in range(depth)])
        self.project = nn.Linear(width, 1)

    def forward(self, q: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, dim = q.shape
        assert dim == self.dim, f"q has dim {dim}, potential expects {self.dim}"

        v = self.lift(q)                                   # (B, H, W, C)
        v = v.permute(0, 3, 1, 2)                          # (B, C, H, W)

        pad = self.reflect_pad
        padded = pad > 0 and H > 2 * pad and W > 2 * pad
        if padded:
            v = F.pad(v, (pad, pad, pad, pad), mode="reflect")

        for block, film in zip(self.blocks, self.films):
            v = block(v)
            if film is not None:
                assert context is not None, "context_dim > 0 requires a context"
                v = film(v, context)

        if padded:
            v = v[..., pad: pad + H, pad: pad + W]
        v = v.permute(0, 2, 3, 1)                          # (B, H, W, C)

        e = self.project(v).squeeze(-1)                    # (B, H, W)
        return e.sum(dim=(-1, -2))                         # (B,)  extensive scalar
