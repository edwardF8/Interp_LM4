"""Adapter: wire a local custom-sized Llama + a trained CLT into circuit-tracer.

circuit-tracer's ReplacementModel.from_pretrained derives the config from a
known HF alias, which does not exist for our custom dims (4 layers, d=384,
vocab=1836).  We instead build the HookedTransformerConfig ourselves (via
clts.tl_model, at TransformerLens's default eps=1e-5 to match how the CLT was
trained/evaluated), construct the replacement subclass directly, load our
weights BEFORE configuration (which wraps mlp/unembed in hooked modules), and
load the CLT with circuit-tracer's own loader so its attribution methods are
available.

Verified against circuit-tracer 0.4.1 source
(clts/.venv-ct/lib/python3.11/site-packages/circuit_tracer/):
  - replacement_model/replacement_model_transformerlens.py
        * _configure_replacement_model(self, transcoder_set)   (line 165)
          reads feature_input_hook / feature_output_hook / scan FROM the
          transcoder object and stores the CLT under `self.transcoders`
          (line 173); the scan id lands on `self.scan` (line 178).
          It also evaluates `"gemma-3" in self.cfg.model_name` (line 170),
          so cfg.model_name must be a non-None string.
        * wraps `block.mlp -> ReplacementMLP(block.mlp)` (old weights move to
          `block.mlp.old_mlp`) and `self.unembed -> ReplacementUnembed(...)`,
          so model weights MUST be loaded before this runs.
        * HookedTransformer.__init__ accepts `tokenizer=` (verified), so the
          direct constructor with tokenizer= is valid.
  - transcoder/cross_layer_transcoder.py
        * load_clt(clt_path, feature_input_hook=, feature_output_hook=,
          scan=, device=, dtype=, lazy_decoder=, lazy_encoder=)  (line 390)
          -- NOTE the kwarg is `scan`, NOT `scan_name` as the design draft
          had it; passing `scan_name=` would raise TypeError.
        * load_clt defaults dtype=bfloat16; we override to fp32 (invariant 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, LlamaForCausalLM
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder.cross_layer_transcoder import (
    CrossLayerTranscoder,
    load_clt,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.tl_model import build_tl_config  # noqa: E402

DEFAULT_ENC_HOOK = "hook_resid_mid"
DEFAULT_DEC_HOOK = "hook_mlp_out"


# ── circuit-tracer bug patch: CrossLayerTranscoder.compute_reconstruction ────
# Upstream sizes the reconstruction by `n_pos = pos_ids.max() + 1` — the highest
# position that has an ACTIVE feature, not the true prompt length. When a prompt's
# last token(s) fire no CLT features (common on sparse / low-layer CLTs like
# grid-L1-H6, and on off-distribution prompt endings), the reconstruction comes
# out one or more positions short, and setup_attribution's
#   error_vectors = mlp_out_cache - reconstruction
# blows up with "tensor a (N) vs b (N-1) at non-singleton dimension 1".
# Fix: take n_pos from `input_acts` (the [n_layers, n_pos, d_model] mlp-in cache,
# always passed during attribution), so the reconstruction always spans the full
# sequence. Feature-less positions get just b_dec (+ skip) — exactly what a
# full-length reconstruction holds there — so the result is unchanged where the
# original worked. We patch the CLASS (not an instance), so it fixes every model
# built this session, including ones already constructed.
def _patched_compute_reconstruction(self, pos_ids, layer_ids, decoder_vectors,
                                    input_acts=None):
    n_pos = input_acts.shape[-2] if input_acts is not None else int(pos_ids.max()) + 1
    flat_idx = layer_ids * n_pos + pos_ids
    recon = torch.zeros(
        n_pos * self.n_layers,
        self.d_model,
        device=decoder_vectors.device,
        dtype=decoder_vectors.dtype,
    ).index_add_(0, flat_idx, decoder_vectors)
    recon = recon.reshape(self.n_layers, n_pos, self.d_model) + self.b_dec[:, None]
    if self.W_skip is not None:
        assert input_acts is not None, (
            "Transcoder has skip connection but no input_acts were provided"
        )
        recon = recon + input_acts @ self.W_skip
    return recon


def _apply_reconstruction_patch():
    """Install the n_pos fix on CrossLayerTranscoder (idempotent)."""
    if getattr(CrossLayerTranscoder.compute_reconstruction, "_npos_patched", False):
        return
    _patched_compute_reconstruction._npos_patched = True
    CrossLayerTranscoder.compute_reconstruction = _patched_compute_reconstruction


_apply_reconstruction_patch()


def _read_hooks(clt_dir: Path) -> tuple[str, str]:
    """Read the CLT's training hooks from its config.yaml (invariant 4).

    feature_input_hook / feature_output_hook must equal the hooks the CLT was
    trained against, or attribution reads/writes the wrong sub-blocks.
    """
    cfg_path = clt_dir / "config.yaml"
    if not cfg_path.exists():
        return DEFAULT_ENC_HOOK, DEFAULT_DEC_HOOK
    cfg = yaml.safe_load(cfg_path.read_text())
    return (
        cfg.get("feature_input_hook", DEFAULT_ENC_HOOK),
        cfg.get("feature_output_hook", DEFAULT_DEC_HOOK),
    )


def load_replacement_model(
    model_dir,
    clt_dir,
    hf_tokenizer_dir,
    scan_name,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TransformerLensReplacementModel:
    """Assemble a circuit-tracer replacement model from local artifacts.

    Args:
        model_dir: local HF Llama checkpoint dir (custom dims).
        clt_dir: dir of the trained CLT (2N safetensors + config.yaml).
        hf_tokenizer_dir: dir loadable by AutoTokenizer.from_pretrained.
        scan_name: single stable scan id, flows CLT -> model -> graph
            (invariant 7).
        device: torch device string.
        dtype: must be fp32 end-to-end (invariant 5); bf16 diverges.
    """
    _apply_reconstruction_patch()   # ensure the n_pos fix is live for this build
    model_dir, clt_dir = Path(model_dir), Path(clt_dir)

    hf_model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(hf_tokenizer_dir))

    # eps is intentionally left at the tl_model default (1e-5); do NOT set it.
    cfg = build_tl_config(hf_model.config, device, dtype)
    cfg.model_name = scan_name                   # invariant 2: non-None string
    cfg.tokenizer_name = str(hf_tokenizer_dir)    # invariant 3: graph-file step

    # invariant 1: load weights BEFORE _configure_replacement_model wraps the
    # mlp/unembed modules (block.mlp -> ReplacementMLP, old weights move under
    # `.old_mlp`); loading after would silently miss those renamed keys.
    model = TransformerLensReplacementModel(cfg, tokenizer=tokenizer)
    model.load_state_dict(convert_llama_weights(hf_model, cfg), strict=False)
    # HookedTransformer.to() takes a single positional device_or_dtype.  dtype
    # is already fp32 throughout (cfg.dtype + fp32 HF source + fp32 converted
    # weights), so only the device move is needed.
    model = model.to(device).eval()

    enc_hook, dec_hook = _read_hooks(clt_dir)     # invariant 4: trained hooks
    clt = load_clt(
        str(clt_dir),
        feature_input_hook=enc_hook,
        feature_output_hook=dec_hook,
        scan=scan_name,                            # invariant 7 (kwarg is `scan`)
        device=torch.device(device),
        dtype=dtype,                               # invariant 5: fp32, NOT bf16
        lazy_decoder=False,
        lazy_encoder=False,
    )
    # _configure_replacement_model reads enc/dec hooks + scan from `clt` and
    # stores it on model.transcoders; it must run AFTER weights are loaded.
    model._configure_replacement_model(clt)
    return model
