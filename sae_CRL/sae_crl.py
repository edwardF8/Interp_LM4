"""SAE_CRL — faithful port of reference/temp-inst-sae/examples/linear_idol_model.py
(LinearIDOL), with two documented paper-side deviations: M = tril(M, -1) (strictly
lower) and TopK on the encoded latents (kept at eval). See the plan's deviations ledger."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file, load_file


def topk_latents(Zp: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k by |value| over the latent axis (dim=1) per (batch, timestep).
    Zp: [batch, z_dim, T]. PAPER deviation P2: TopK on encoded latents (all timesteps)."""
    if k <= 0 or k >= Zp.shape[1]:
        return Zp
    idx = Zp.abs().topk(k, dim=1).indices
    mask = torch.zeros_like(Zp)
    mask.scatter_(1, idx, 1.0)
    return Zp * mask


class SAE_CRL(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, tau: int,
                 w: float = 0.5, noise_mode: str = "lap", topk_sparsity: int = 100):
        super().__init__()
        self.x_dim, self.z_dim, self.tau = x_dim, z_dim, tau
        self.w, self.noise_mode, self.topk_sparsity = w, noise_mode, topk_sparsity
        # per-lag time-delayed matrices, ZERO init (ref lines 25-29, NOT xavier)
        self.Bs = nn.ParameterList([nn.Parameter(torch.zeros(z_dim, z_dim)) for _ in range(tau)])
        self.F_enc = nn.Parameter(torch.ones(x_dim, z_dim))   # ref line 31
        self.F_dec = nn.Parameter(torch.ones(z_dim, x_dim))   # ref line 33
        self.M = nn.Parameter(torch.ones(z_dim, z_dim))       # ref line 36
        self.init_params()
        self._hook_name = None

    def init_params(self):  # ref lines 50-53
        nn.init.xavier_normal_(self.F_enc.data)
        nn.init.xavier_normal_(self.F_dec.data)
        nn.init.xavier_normal_(self.M.data)

    def forward(self, Xp: torch.Tensor, enable_w: bool = False):
        """Xp: [batch, x_dim, tau+1], last index = current token. Returns the
        reference 6-tuple (mse_Xt, mse_Zt, indep, sparse_Bs, sparse_M, sparse_Zt)."""
        # Encode all timesteps (ref line 78)
        Zp = torch.einsum("hd,bdt->bht", self.F_enc.T, Xp)        # [batch, z_dim, tau+1]
        # P2: TopK on encoded latents (all timesteps), KEPT ON at eval
        if self.topk_sparsity > 0:
            Zp = topk_latents(Zp, self.topk_sparsity)
        # Decode all; MSE on LAST position only (ref lines 79-80)
        recons = torch.einsum("dh,bht->bdt", self.F_dec.T, Zp)
        loss_mse_Xt = F.mse_loss(recons[:, :, -1], Xp[:, :, -1])

        M = torch.tril(self.M, diagonal=-1)                       # P1: strictly lower
        if enable_w:
            w, _w = self.w, 1.0 - self.w
        else:
            w, _w = 1.0, 1.0
        Zt = _w * torch.einsum("hd,bd->bh", M, Zp[:, :, self.tau])   # M @ z_t (ref line 91)
        loss_sparse_Bs = 0.0
        for lag in range(1, self.tau + 1):                        # ref lines 92-96
            B_lag = self.Bs[lag - 1]
            loss_sparse_Bs = loss_sparse_Bs + F.l1_loss(B_lag, torch.zeros_like(B_lag))
            Zt = Zt + w * torch.einsum("hd,bd->bh", B_lag, Zp[:, :, self.tau - lag])

        loss_mse_Zt = F.mse_loss(Zt, Zp[:, :, self.tau])          # ref line 106
        Et = Zp[:, :, self.tau] - Zt                              # ref line 109
        if self.noise_mode == "gau":
            loss_indep = torch.trace(torch.cov(Et))               # ref lines 110-112
        elif self.noise_mode == "lap":
            loss_indep = F.l1_loss(Et, torch.zeros_like(Et))      # ref lines 113-115
        else:
            raise NotImplementedError(self.noise_mode)
        loss_sparse_M = F.l1_loss(M, torch.zeros_like(M))         # ref line 121
        loss_sparse_Zt = F.l1_loss(Zt, torch.zeros_like(Zt))      # ref line 124
        return (loss_mse_Xt, loss_mse_Zt, loss_indep,
                loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt)

    @torch.no_grad()
    def aggB(self) -> torch.Tensor:
        """Max-pool |B_lag| over lags -> [z_dim, z_dim] lag-agnostic edge map (additive S5)."""
        if self.tau == 0:
            return torch.zeros(self.z_dim, self.z_dim, device=self.M.device)
        return torch.stack([b.abs() for b in self.Bs], dim=0).amax(dim=0)

    def save_to_dir(self, out_dir, model_name: str, hook_name: str, layer: int) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        tensors = {"F_enc": self.F_enc.detach().cpu(), "F_dec": self.F_dec.detach().cpu(),
                   "M": self.M.detach().cpu()}
        for i, b in enumerate(self.Bs):
            tensors[f"B_{i}"] = b.detach().cpu()
        save_file(tensors, str(out_dir / "sae_crl.safetensors"))
        cfg = {"model_name": model_name, "hook_name": hook_name, "layer": layer,
               "x_dim": self.x_dim, "z_dim": self.z_dim, "tau": self.tau, "w": self.w,
               "noise_mode": self.noise_mode, "topk_sparsity": self.topk_sparsity}
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    @classmethod
    def load_from_dir(cls, in_dir) -> "SAE_CRL":
        in_dir = Path(in_dir)
        with open(in_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        m = cls(x_dim=cfg["x_dim"], z_dim=cfg["z_dim"], tau=cfg["tau"], w=cfg.get("w", 0.5),
                noise_mode=cfg.get("noise_mode", "lap"), topk_sparsity=cfg.get("topk_sparsity", 100))
        t = load_file(str(in_dir / "sae_crl.safetensors"))
        with torch.no_grad():
            m.F_enc.copy_(t["F_enc"]); m.F_dec.copy_(t["F_dec"]); m.M.copy_(t["M"])
            for i in range(cfg["tau"]):
                m.Bs[i].copy_(t[f"B_{i}"])
        m._hook_name = cfg["hook_name"]
        return m
