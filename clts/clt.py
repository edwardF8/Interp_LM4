"""Cross-Layer Transcoder.

Matches Anthropic circuit-tracer's CrossLayerTranscoder packed-tensor
layout so trained weights load into their tooling without conversion.

Shapes (N = n_layers, D = d_model, d_t = d_transcoder = expansion * D):
    W_enc:     [N, d_t, D]
    b_enc:     [N, d_t]
    threshold: [N, d_t]
    W_dec:     ParameterList of length N; entry i is [d_t, N - i, D]
    b_dec:     [N, D]
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file, load_file


JUMPRELU_INIT_THRESHOLD = 0.1
JUMPRELU_BANDWIDTH = 2.0


class JumpReLU(torch.autograd.Function):
    """JumpReLU with rectangle-kernel STE for threshold gradient.

    Forward: y = x * 1[x > threshold]
    Backward (x):         pass-through where x > threshold
    Backward (threshold): -threshold/bandwidth * 1[|x - threshold| < bandwidth/2]
    Standard recipe from Rajamanoharan et al. (2024).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bw = ctx.bandwidth
        x_grad = grad_output * (x > threshold).to(x.dtype)
        in_window = ((x - threshold).abs() < bw / 2).to(x.dtype)
        threshold_grad = grad_output * (-threshold / bw) * in_window
        # Sum threshold grad across batch axes; threshold is [d_t], grad is [B, d_t]
        while threshold_grad.dim() > threshold.dim():
            threshold_grad = threshold_grad.sum(0)
        return x_grad, threshold_grad, None


class CrossLayerTranscoder(nn.Module):
    DEFAULT_ENC_HOOK = "hook_resid_mid"
    DEFAULT_DEC_HOOK = "hook_mlp_out"

    def __init__(self, n_layers: int, d_model: int, expansion: int = 16):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_transcoder = expansion * d_model

        N, D, d_t = n_layers, d_model, self.d_transcoder

        self.W_enc = nn.Parameter(torch.empty(N, d_t, D))
        self.b_enc = nn.Parameter(torch.zeros(N, d_t))
        self.threshold = nn.Parameter(torch.full((N, d_t), JUMPRELU_INIT_THRESHOLD))
        self.b_dec = nn.Parameter(torch.zeros(N, D))

        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.empty(d_t, N - i, D)) for i in range(N)
        ]) #One for each layer

        # Kaiming-uniform init for encoder/decoder weights (matches typical
        # sae_lens init; avoids any param staying at zero from above).
        for i in range(N):
            nn.init.kaiming_uniform_(self.W_enc[i], a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.W_dec[i], a=5 ** 0.5)

    def encode(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """For each source laeyer L, compute read in inputs from the
        residual stream,a[L] = JumpReLU(W_enc[L] x[L] + b_enc[L])."""
        assert len(x_list) == self.n_layers, \
            f"expected {self.n_layers} inputs, got {len(x_list)}"
        a_list = []
        for L in range(self.n_layers):
            preact = x_list[L] @ self.W_enc[L].T + self.b_enc[L]
            a = JumpReLU.apply(preact, self.threshold[L], JUMPRELU_BANDWIDTH)
            a_list.append(a)
        return a_list

    def decode(self, a_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """For each target L', sum decoder contributions from sources L <= L'.
        For each layer,
            yhat = bias for L' + summation W_dec from L -> L' @ a_list[L]
            yhat = bias for L' + summation Decoder from L -> L' @ post-encoder from L
        """
        y_hat_list = []
        batch_size = a_list[0].shape[0]
        for L_prime in range(self.n_layers):
            y_hat = self.b_dec[L_prime].unsqueeze(0).expand(batch_size, -1).clone() #add bias
            for L in range(L_prime + 1):
                target_idx = L_prime - L
                dec_slice = self.W_dec[L][:, target_idx, :]  # [d_t, D]
                y_hat = y_hat + a_list[L] @ dec_slice
            y_hat_list.append(y_hat)
        return y_hat_list

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.decode(self.encode(x_list))

    def _decoder_norms(self) -> list[torch.Tensor]:
        """Per-feature L2 norm of the *summed* decoder vector across all
        downstream targets. Shape: list of N tensors, each [d_t].

        For source L, sums the decoder contributions across the N - L
        targets, then takes the L2 norm per feature.
        """
        norms = []
        for L in range(self.n_layers):
            summed = self.W_dec[L].sum(dim=1)             # [d_t, D]
            norms.append(summed.norm(dim=-1))             # [d_t]
        return norms

    def compute_loss(
        self,
        x_list: list[torch.Tensor],
        y_list: list[torch.Tensor],
        l0_coefficient: float,
        tanh_scale: float = 4.0,
        pre_act_coef: float = 3e-6,
    ) -> dict[str, torch.Tensor]:
        """Joint multi-layer CLT loss.

        Returns a dict with keys: total, recon, sparsity, preact, plus
        per-layer recon_L{i} and l0_L{i} for logging.
        """
        # Encoder pre-activations (needed for both encode output and preact loss).
        preacts = [
            x_list[L] @ self.W_enc[L].T + self.b_enc[L]
            for L in range(self.n_layers)
        ]
        a_list = [
            JumpReLU.apply(preacts[L], self.threshold[L], JUMPRELU_BANDWIDTH)
            for L in range(self.n_layers)
        ]
        y_hat_list = self.decode(a_list)

        # Per-layer reconstruction MSE (mean over batch and features).
        recon_per_layer = [
            (y_hat_list[L] - y_list[L]).pow(2).mean()
            for L in range(self.n_layers)
        ]
        recon = sum(recon_per_layer) / self.n_layers

        # Decoder-norm-weighted tanh sparsity penalty.
        dec_norms = self._decoder_norms()
        sparsity_terms = []
        l0_per_layer = []
        for L in range(self.n_layers):
            weighted = a_list[L] * dec_norms[L].unsqueeze(0)      # [B, d_t]
            sparsity_terms.append(torch.tanh(tanh_scale * weighted).sum(dim=-1).mean())
            l0_per_layer.append((a_list[L] > 0).float().sum(dim=-1).mean())
        sparsity = l0_coefficient * sum(sparsity_terms)

        # Pre-activation L2 penalty (matches sae_lens preact loss intent).
        preact = pre_act_coef * sum(p.pow(2).mean() for p in preacts)

        total = recon + sparsity + preact

        out = {"total": total, "recon": recon, "sparsity": sparsity, "preact": preact}
        for L in range(self.n_layers):
            out[f"recon_L{L}"] = recon_per_layer[L].detach()
            out[f"l0_L{L}"] = l0_per_layer[L].detach()
        return out

    def save_to_dir(
        self,
        out_dir: str | Path,
        model_name: str,
        feature_input_hook: str = DEFAULT_ENC_HOOK,
        feature_output_hook: str = DEFAULT_DEC_HOOK,
    ) -> None:
        """Save as 2N safetensors + config.yaml in circuit-tracer's exact format."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.n_layers):
            save_file({
                f"W_enc_{i}":     self.W_enc.data[i].contiguous(),
                f"b_enc_{i}":     self.b_enc.data[i].contiguous(),
                f"b_dec_{i}":     self.b_dec.data[i].contiguous(),
                f"threshold_{i}": self.threshold.data[i].contiguous(),
            }, str(out_dir / f"W_enc_{i}.safetensors"))
            save_file({
                f"W_dec_{i}": self.W_dec[i].data.contiguous(),
            }, str(out_dir / f"W_dec_{i}.safetensors"))

        cfg = {
            "model_name": model_name,
            "model_kind": "cross_layer_transcoder",
            "feature_input_hook": feature_input_hook,
            "feature_output_hook": feature_output_hook,
        }
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    @classmethod
    def load_from_dir(cls, in_dir: str | Path) -> "CrossLayerTranscoder":
        """Reconstruct a CLT from the 2N safetensors layout. Dimensions are
        inferred from tensor shapes (matches circuit-tracer's loader)."""
        in_dir = Path(in_dir)

        # Discover N from the number of W_enc_* files.
        enc_files = sorted(in_dir.glob("W_enc_*.safetensors"))
        n_layers = len(enc_files)
        assert n_layers > 0, f"no W_enc_*.safetensors found in {in_dir}"

        # Peek at W_enc_0 to infer d_t and D.
        enc0 = load_file(str(in_dir / "W_enc_0.safetensors"))
        d_t, D = enc0["W_enc_0"].shape
        assert d_t % D == 0, f"d_transcoder {d_t} not a multiple of d_model {D}"
        expansion = d_t // D

        clt = cls(n_layers=n_layers, d_model=D, expansion=expansion)
        with torch.no_grad():
            for i in range(n_layers):
                enc = load_file(str(in_dir / f"W_enc_{i}.safetensors"))
                dec = load_file(str(in_dir / f"W_dec_{i}.safetensors"))
                clt.W_enc.data[i].copy_(enc[f"W_enc_{i}"])
                clt.b_enc.data[i].copy_(enc[f"b_enc_{i}"])
                clt.b_dec.data[i].copy_(enc[f"b_dec_{i}"])
                clt.threshold.data[i].copy_(enc[f"threshold_{i}"])
                clt.W_dec[i].data.copy_(dec[f"W_dec_{i}"])
        return clt
