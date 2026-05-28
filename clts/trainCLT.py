"""Train cross-layer transcoders on a base Llama checkpoint.

Single run, default hooks:
    python clts/trainCLT.py --model-dir <path> --data-dir <path>

Sweep over (expansion x l0 x lr):
    python clts/trainCLT.py --model-dir <path> --data-dir <path> --sweep

Outputs land in:
    STORAGE_ROOT / clt_runs / <model-name> / [sweep-<id>|standalone] / <trial> / final/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore

# Project imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clts.clt import CrossLayerTranscoder  # noqa: E402
from clts.evalCLT import (  # noqa: E402
    capture_activations, compute_layer_metrics,
    ce_recovered_full, ce_recovered_per_layer,
)
from clts.export_tokenizer import ensure_hf_tokenizer  # noqa: E402
from util.bio_sampler import BioSampler  # noqa: E402
from util.condensed_tokenizer import CondensedTokenizer  # noqa: E402
from util.diverse_subset import DiverseBioSubset  # noqa: E402


# ============================================================================
# Output location — edit STORAGE_ROOT if you move workspaces.
# ============================================================================
STORAGE_ROOT = Path(os.environ.get(
    "CLT_STORAGE_ROOT",
    "/jet/home/friedmae/data_storage/LM4_Results",
))


# ============================================================================
# Defaults — overridable via CLI flags or the sweep grid.
# ============================================================================
DEFAULTS = {
    "n_examples":     10_000,
    "epochs":         30,
    "context_size":   512,
    "expansion":      16,
    "l0_coefficient": 5.0,
    "lr":             5e-5,
}
CLT_SEED = 0
BATCH_SIZE = 4096


# ============================================================================
# Module state (set by setup(), reused across sweep trials).
# ============================================================================
ARGS: argparse.Namespace | None = None
device: str | None = None
model: HookedTransformer | None = None
tokenizer: CondensedTokenizer | None = None
hf_tokenizer_path: Path | None = None
sampler: BioSampler | None = None
eval_tokens: torch.Tensor | None = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup(args: argparse.Namespace) -> None:
    """Load model + tokenizer + sampler + held-out eval tokens. Called once
    per process; sweep trials reuse the globals."""
    global ARGS, device, model, tokenizer, hf_tokenizer_path, sampler, eval_tokens
    ARGS = args

    # Pre-flight: STORAGE_ROOT must be writable.
    try:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        probe = STORAGE_ROOT / f".write_probe.{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError as e:
        raise SystemExit(
            f"STORAGE_ROOT is not writable: {STORAGE_ROOT}\n  {e}\n"
            f"Set CLT_STORAGE_ROOT env var or edit STORAGE_ROOT at top of trainCLT.py."
        )
    print(f"[storage] {STORAGE_ROOT}  (writable)")

    if args.model_name is None:
        args.model_name = (
            args.model_dir.parent.name if args.model_dir.name == "final"
            else args.model_dir.name
        )

    device = pick_device()
    dtype = torch.float32

    hf_model = LlamaForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).eval()
    hf_cfg = hf_model.config
    tl_cfg = HookedTransformerConfig(
        n_layers=hf_cfg.num_hidden_layers,
        d_model=hf_cfg.hidden_size,
        d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        n_heads=hf_cfg.num_attention_heads,
        d_mlp=hf_cfg.intermediate_size,
        d_vocab=hf_cfg.vocab_size,
        n_ctx=hf_cfg.max_position_embeddings,
        act_fn="silu",
        normalization_type="RMS",
        gated_mlp=True,
        positional_embedding_type="rotary",
        rotary_base=int(getattr(hf_cfg, "rope_theta", 10000.0)),
        rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        final_rms=True,
        tie_word_embeddings=hf_cfg.tie_word_embeddings,
        initializer_range=hf_cfg.initializer_range,
        n_key_value_heads=hf_cfg.num_key_value_heads,
        device=device,
    )
    model = HookedTransformer(tl_cfg)
    model.load_state_dict(convert_llama_weights(hf_model, tl_cfg), strict=False)
    model.to(device).eval()
    print(f"[model]   {args.model_dir} (name: {args.model_name})")
    print(f"          n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
          f"n_heads={model.cfg.n_heads}, d_vocab={model.cfg.d_vocab}")

    tokenizer = CondensedTokenizer.from_remap_path(args.data_dir / "old_to_new.json")
    hf_tokenizer_path = ensure_hf_tokenizer(args.data_dir)
    print(f"[tokenizer] {hf_tokenizer_path}")

    sampler = BioSampler(args.data_dir / "people.json", fields=("birthday",), seed=CLT_SEED)

    # Held-out eval slice (seed+1, matches trainSAE.py convention).
    eval_subset = DiverseBioSubset(
        sampler, tokenizer, context_size=args.context_size, seed=CLT_SEED + 1
    )
    eval_rows = eval_subset.to_hf_dataset(64, verbose=False)["input_ids"]
    eval_tokens = torch.tensor(np.array(eval_rows), dtype=torch.long, device=device)
    print(f"[data]    {args.data_dir}")
    print(f"          {len(sampler.people):,} people, {sampler.n_templates} templates, "
          f"eval tokens: {tuple(eval_tokens.shape)}")

    # Section 2 coverage check: exposures per person at current --n-examples.
    train_subset = DiverseBioSubset(sampler, tokenizer, context_size=args.context_size, seed=CLT_SEED)
    rows = train_subset.to_hf_dataset(args.n_examples, verbose=False)["input_ids"]
    rows_np = np.array(rows)
    n_bios = int((rows_np == tokenizer.eos_token_id).sum())
    exposures = n_bios / max(1, len(sampler.people))
    expected_missing = len(sampler.people) * pow(2.71828, -exposures)
    print(f"[coverage] n_examples={args.n_examples} -> ~{n_bios:,} bios/epoch")
    print(f"           ~{exposures:.1f} exposures/person/epoch")
    print(f"           expected people with 0 exposures: ~{int(expected_missing):,} "
          f"({100*expected_missing/len(sampler.people):.1f}%)")


# ============================================================================
# Per-trial training (called by wandb.agent OR directly for single runs)
# ============================================================================

def trial_name(expansion: int, l0_coefficient: float, lr: float,
               epochs: int, n_examples: int) -> str:
    return f"mult{expansion}_l0{l0_coefficient:g}_lr{lr:g}_ep{epochs}_n{n_examples}"


def train_one_run(wandb_config_override: dict | None = None) -> None:
    """One CLT training run end-to-end. wandb_config_override comes from
    wandb.agent during a sweep; None for standalone runs."""
    import wandb

    wandb.init(project="interpLM4")
    run_id, run_entity, run_project = wandb.run.id, wandb.run.entity, wandb.run.project
    sweep_id = wandb.run.sweep_id
    cfg = wandb.config

    expansion      = cfg.get("expansion",      ARGS.expansion)
    l0_coefficient = cfg.get("l0_coefficient", ARGS.l0_coefficient)
    lr             = cfg.get("lr",             ARGS.lr)
    epochs         = cfg.get("epochs",         ARGS.epochs)
    n_examples     = cfg.get("n_examples",     ARGS.n_examples)
    context_size   = cfg.get("context_size",   ARGS.context_size)

    name = trial_name(expansion, l0_coefficient, lr, epochs, n_examples)
    sweep_folder = f"sweep-{sweep_id}" if sweep_id else "standalone"
    run_dir = STORAGE_ROOT / "clt_runs" / ARGS.model_name / sweep_folder / name
    final_dir = run_dir / "final"

    # Build training data.
    train_subset = DiverseBioSubset(
        sampler, tokenizer, context_size=context_size, seed=CLT_SEED
    )
    train_rows = train_subset.to_hf_dataset(n_examples, verbose=False)["input_ids"]
    train_tokens = torch.tensor(np.array(train_rows), dtype=torch.long, device=device)
    n_tokens = train_subset.n_tokens(n_examples)
    total_steps = (epochs * n_tokens) // BATCH_SIZE
    l0_warmup = total_steps // 10
    lr_warmup = total_steps // 50
    print(f"[trial]   {name}")
    print(f"          training tokens={n_tokens:,}, steps={total_steps:,}, "
          f"l0_warmup={l0_warmup}, lr_warmup={lr_warmup}")

    # Build CLT.
    clt = CrossLayerTranscoder(
        n_layers=model.cfg.n_layers, d_model=model.cfg.d_model, expansion=expansion,
    ).to(device)
    opt = torch.optim.Adam(clt.parameters(), lr=lr, betas=(0.9, 0.999))

    # Token-batch iterator. Yields [B, T] slices, looping over training_tokens.
    rows_per_batch = max(1, BATCH_SIZE // context_size)
    def token_batches():
        n_rows = train_tokens.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n_rows)
            for start in range(0, n_rows - rows_per_batch + 1, rows_per_batch):
                yield train_tokens[perm[start:start + rows_per_batch]]

    step = 0
    LOG_EVERY = 30
    EVAL_EVERY = 600

    for batch_tokens in token_batches():
        # Capture activations from the frozen base model.
        x_list, y_list = capture_activations(
            model, batch_tokens,
            enc_hook_template=ARGS.enc_hook_template,
            dec_hook_template=ARGS.dec_hook_template,
        )

        # L0 warmup: linear ramp of sparsity coefficient.
        ramp = min(1.0, (step + 1) / max(1, l0_warmup))
        lam = l0_coefficient * ramp

        # LR warmup: linear ramp from 0 to lr over lr_warmup steps.
        lr_ramp = min(1.0, (step + 1) / max(1, lr_warmup))
        for g in opt.param_groups:
            g["lr"] = lr * lr_ramp

        losses = clt.compute_loss(x_list, y_list, l0_coefficient=lam)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

        if step % LOG_EVERY == 0:
            log_payload = {"clt_train/mse_total": losses["recon"].item(),
                           "clt_train/sparsity_loss": losses["sparsity"].item(),
                           "clt_train/preact_loss": losses["preact"].item(),
                           "clt_train/l0_coef_effective": lam,
                           "clt_train/lr": opt.param_groups[0]["lr"]}
            for L in range(model.cfg.n_layers):
                log_payload[f"clt_train/mse_L{L}"] = losses[f"recon_L{L}"].item()
                log_payload[f"clt_train/l0_L{L}"] = losses[f"l0_L{L}"].item()
            wandb.log(log_payload, step=step)

        if step > 0 and step % EVAL_EVERY == 0:
            x_eval, y_eval = capture_activations(
                model, eval_tokens,
                enc_hook_template=ARGS.enc_hook_template,
                dec_hook_template=ARGS.dec_hook_template,
            )
            metrics = compute_layer_metrics(clt, x_eval, y_eval)
            ce = ce_recovered_full(
                model, clt, eval_tokens,
                enc_hook_template=ARGS.enc_hook_template,
                dec_hook_template=ARGS.dec_hook_template,
            )
            wandb.log(
                {f"clt_eval/{k}": v for k, v in {**metrics, **ce}.items()},
                step=step,
            )

        step += 1

    # Final eval + save.
    x_eval, y_eval = capture_activations(
        model, eval_tokens,
        enc_hook_template=ARGS.enc_hook_template,
        dec_hook_template=ARGS.dec_hook_template,
    )
    final_metrics = compute_layer_metrics(clt, x_eval, y_eval)
    final_ce = ce_recovered_full(
        model, clt, eval_tokens,
        enc_hook_template=ARGS.enc_hook_template,
        dec_hook_template=ARGS.dec_hook_template,
    )
    final_per_layer = ce_recovered_per_layer(
        model, clt, eval_tokens,
        enc_hook_template=ARGS.enc_hook_template,
        dec_hook_template=ARGS.dec_hook_template,
    )

    clt.save_to_dir(final_dir, model_name=ARGS.model_name,
                    feature_input_hook=ARGS.enc_hook_template.split(".")[-1],
                    feature_output_hook=ARGS.dec_hook_template.split(".")[-1])

    payload = {f"final_eval/{k}": v for k, v in
               {**final_metrics, **final_ce, **final_per_layer}.items()}
    payload["storage_path"] = str(final_dir)
    payload["tokenizer_path"] = str(hf_tokenizer_path)
    wandb.log(payload)
    wandb.run.summary.update(payload)
    print(f"[final]   saved to {final_dir}")
    print(f"          ce_recovered={final_ce['ce_recovered']:.4f}")
    wandb.finish()


# ============================================================================
# Sweep config
# ============================================================================

def build_sweep_config() -> dict:
    return {
        "program": "trainCLT.py",
        "method":  "grid",
        "name":    f"clt_sweep_{ARGS.model_name}",
        "metric":  {"name": "final_eval/ce_recovered", "goal": "maximize"},
        "parameters": {
            "expansion":      {"values": [8, 16]},
            "l0_coefficient": {"values": [2.0, 5.0, 10.0]},
            "lr":             {"values": [3e-5, 1e-4]},
            "epochs":         {"value":  50},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
    }


def _patch_signal_for_worker_threads():
    """wandb.agent runs trials in worker threads; signal.signal() rejects
    non-main-thread calls. Wrap signal.signal so it no-ops off the main thread."""
    import signal
    import threading
    _real = signal.signal

    def _safe(signum, handler):
        if threading.current_thread() is threading.main_thread():
            return _real(signum, handler)
        return None
    signal.signal = _safe


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HF Llama checkpoint dir (the base model the CLT attaches to).")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dataset dir with people.json + old_to_new.json.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Identifier for this base model in output paths. "
                        "Default: parent dir of --model-dir.")

    p.add_argument("--enc-hook-template", type=str,
                   default="blocks.{layer}.hook_resid_mid",
                   help="Encoder input hook template (must contain '{layer}').")
    p.add_argument("--dec-hook-template", type=str,
                   default="blocks.{layer}.hook_mlp_out",
                   help="Decoder target hook template (must contain '{layer}').")

    p.add_argument("--expansion", type=int, default=DEFAULTS["expansion"],
                   help="d_transcoder = expansion * d_model.")
    p.add_argument("--l0", dest="l0_coefficient", type=float,
                   default=DEFAULTS["l0_coefficient"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--context-size", type=int, default=DEFAULTS["context_size"])
    p.add_argument("--n-examples", type=int, default=DEFAULTS["n_examples"])

    # NOTE: --eval-mode flag deferred. Current pipeline always runs the "quick"
    # eval (64 rows) + per-layer CE-recovered diagnostic at final time. The
    # spec's "full" mode (per-person / per-template breakdowns over the full
    # 50k-people sample) is documented as a follow-up; the existing single
    # ce_recovered headline number is sufficient for sweep selection.

    p.add_argument("--sweep", action="store_true",
                   help="Launch wandb grid sweep over (expansion x l0 x lr).")

    return p.parse_args()


def main():
    import wandb
    args = parse_args()
    setup(args)
    if args.sweep:
        _patch_signal_for_worker_threads()
        cfg = build_sweep_config()
        sweep_id = wandb.sweep(cfg, project="interpLM4")
        print(f"[sweep]   registered: {sweep_id}")
        wandb.agent(sweep_id, function=train_one_run)
    else:
        train_one_run()


if __name__ == "__main__":
    main()
