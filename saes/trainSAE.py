"""Train SAEs on a base model.

Single run, default hook ('blocks.1.hook_mlp_out'):
    python saes/trainSAE.py --model-dir <path> --data-dir <path>

Single run on a specific layer:
    python saes/trainSAE.py --model-dir <path> --data-dir <path> --hook "blocks.3.hook_mlp_out"

Multi-trial sweep over layers + hyperparams:
    python saes/trainSAE.py --model-dir <path> --data-dir <path> \\
        --layers 2,4,6 --hook-template "blocks.{layer}.hook_mlp_out" --sweep

Outputs land in:
    saes/sae_runs/<model-name>/[sweep-<id>|standalone]/<trial>/final/

<trial> is L<layer>_mult<m>_l0<x>_lr<r>_ep<e>_n<k>, so different-layer trials
never collide. <model-name> defaults to the parent dir of --model-dir (e.g.
'/.../grid-L8-H6/final' -> 'grid-L8-H6').
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from sae_lens import (
    HookedSAETransformer,
    JumpReLUTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    SAE,
)
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights  # type: ignore
import wandb

from saes.evalSAE import print_report, sae_eval
from util.bio_sampler import BioSampler
from util.condensed_tokenizer import CondensedTokenizer
from util.diverse_subset import DiverseBioSubset  # type: ignore


# ============================================================================
# OUTPUT LOCATION   ← edit STORAGE_ROOT if you move workspaces
# ----------------------------------------------------------------------------
# Trained SAEs land permanently under STORAGE_ROOT. On PSC this MUST be on
# Ocean (data_storage) — $HOME has a ~25 GB quota that fills up fast.
#
# With n_checkpoints=0 sae_lens writes exactly one ~20 MB safetensors file
# per trial, at the end of runner.run(). That's a single large sequential
# write — exactly Ocean's strong case per PSC's docs — so we write directly
# here, no $LOCAL staging.
#
# If you ever raise n_checkpoints to save mid-training snapshots, switch
# output_path to a $LOCAL staging dir and copy to STORAGE_ROOT after eval
# (many small writes on Ocean is slow + chews through inode quota).
# ============================================================================
STORAGE_ROOT = Path("/jet/home/friedmae/data_storage/LM4_Results/saes")


# ============================================================================
# Defaults — overridable via CLI flags or the sweep grid.
# ============================================================================

DEFAULTS = {
    "n_examples":     10_000,
    "epochs":         30,
    "context_size":   512,
    "sae_mult":       8,
    "l0_coefficient": 5.0,
    "lr":             5e-5,
}

SAE_seed = 0
batch_size = 4096


# ============================================================================
# Module state — populated once by `setup()`, reused by every sweep trial
# (wandb.agent reuses this Python process).
# ============================================================================

ARGS: argparse.Namespace | None = None
device: str | None = None
model: HookedTransformer | None = None
tokenizer: CondensedTokenizer | None = None
sampler: BioSampler | None = None
eval_tokens: torch.Tensor | None = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup(args: argparse.Namespace) -> None:
    """One-time module setup: model, tokenizer, sampler, held-out eval tokens.

    Populates module globals. Called once per process; every trial of a sweep
    then reads from those globals without reloading.
    """
    global ARGS, device, model, tokenizer, sampler, eval_tokens
    ARGS = args

    # Pre-flight: STORAGE_ROOT must be writable. Catches permission / typo /
    # quota bugs in <1s, instead of training for 30 minutes and dying when
    # sae_lens tries to save the final SAE.
    try:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        probe = STORAGE_ROOT / f".write_probe.{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError as e:
        raise SystemExit(
            f"STORAGE_ROOT is not writable: {STORAGE_ROOT}\n  {e}\n"
            f"Edit STORAGE_ROOT at the top of saes/trainSAE.py."
        )
    print(f"[storage] {STORAGE_ROOT}  (writable)")

    if args.model_name is None:
        # '/jet/.../grid-L8-H6/final' -> 'grid-L8-H6'
        # '/jet/.../grid-L8-H6'       -> 'grid-L8-H6'
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

    remap_path = args.data_dir / "old_to_new.json"
    tokenizer = CondensedTokenizer.from_remap_path(remap_path)
    sampler = BioSampler(args.data_dir / "people.json", fields=("birthday",), seed=SAE_seed)

    eval_subset = DiverseBioSubset(
        sampler, tokenizer, context_size=args.context_size, seed=SAE_seed + 1
    )
    eval_rows = eval_subset.to_hf_dataset(64, verbose=False)["input_ids"]
    eval_tokens = torch.tensor(np.array(eval_rows), dtype=torch.long, device=device)
    print(f"[data]    {args.data_dir}")
    print(f"          {len(sampler.people):,} people, {sampler.n_templates} templates, "
          f"eval tokens: {tuple(eval_tokens.shape)}")


# ============================================================================
# Naming helpers
# ============================================================================

_LAYER_RE = re.compile(r"\.(\d+)\.")


def layer_from_hook(hook_name: str) -> int | None:
    """'blocks.3.hook_mlp_out' -> 3. Returns None if hook has no numeric segment."""
    m = _LAYER_RE.search(hook_name)
    return int(m.group(1)) if m else None


def trial_name(hook_name: str, sae_mult: int, l0_coefficient: float, lr: float,
               epochs: int, n_examples: int) -> str:
    """Per-trial directory name. Prefixed with L<n> when the hook has a layer
    index so different-layer trials never collide.
    """
    base = f"mult{sae_mult}_l0{l0_coefficient:g}_lr{lr:g}_ep{epochs}_n{n_examples}"
    layer = layer_from_hook(hook_name)
    return f"L{layer}_{base}" if layer is not None else base


def build_sweep_config(args: argparse.Namespace, hook_name: str) -> dict:
    """Build a wandb sweep grid for *one* layer / hook.

    Per-layer sweeps (rather than one combined sweep with hook_name as an
    axis) give cleaner per-layer rankings on the wandb UI and let Hyperband
    prune trials within a layer rather than cross-layer.
    """
    layer = layer_from_hook(hook_name)
    layer_tag = f"L{layer}" if layer is not None else hook_name.replace(".", "_")
    return {
        "program": "trainSAE.py",
        "method":  "grid",
        "name":    f"sae_sweep_{args.model_name}_{layer_tag}",
        "metric":  {"name": "final_eval/ce_recovered", "goal": "maximize"},
        "parameters": {
            "hook_name":      {"value":  hook_name},   # singleton — baked into config
            "l0_coefficient": {"values": [2.0, 5.0, 10.0]},
            "sae_mult":       {"values": [8, 16]},
            "lr":             {"values": [3e-5, 1e-4]},
            "epochs":         {"value":  50},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
    }


def expand_layers(args: argparse.Namespace) -> list[str]:
    """Parse --layers into hook names via --hook-template."""
    if not args.layers:
        raise SystemExit("--sweep requires --layers (e.g. --layers 2,4,6)")
    if "{layer}" not in args.hook_template:
        raise SystemExit("--hook-template must contain '{layer}'")
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    return [args.hook_template.format(layer=L) for L in layers]


# ============================================================================
# Per-trial training (called by wandb.agent OR directly for single runs)
# ============================================================================

def train_one_run() -> None:
    """One SAE training run + held-out eval, logged to the current wandb run.

    Reads swept config from wandb when present; otherwise falls back to ARGS.
    """
    wandb.init(project="interpLM4")
    run_id      = wandb.run.id
    run_entity  = wandb.run.entity
    run_project = wandb.run.project
    sweep_id    = wandb.run.sweep_id   # None when this isn't a sweep trial
    sweep_cfg   = wandb.config

    hook_name      = sweep_cfg.get("hook_name",      ARGS.hook)
    n_examples     = sweep_cfg.get("n_examples",     ARGS.n_examples)
    epochs         = sweep_cfg.get("epochs",         ARGS.epochs)
    context_size   = sweep_cfg.get("context_size",   ARGS.context_size)
    sae_mult       = sweep_cfg.get("sae_mult",       ARGS.sae_mult)
    l0_coefficient = sweep_cfg.get("l0_coefficient", ARGS.l0_coefficient)
    lr             = sweep_cfg.get("lr",             ARGS.lr)

    sae_name        = trial_name(hook_name, sae_mult, l0_coefficient, lr, epochs, n_examples)
    sweep_folder    = f"sweep-{sweep_id}" if sweep_id else "standalone"
    SAE_RUN_DIR     = STORAGE_ROOT / "sae_runs" / ARGS.model_name / sweep_folder / sae_name
    checkpoint_path = str(SAE_RUN_DIR / "checkpoints")
    output_path     = str(SAE_RUN_DIR / "final")

    subset = DiverseBioSubset(sampler, tokenizer, context_size=context_size, seed=SAE_seed)
    sae_dataset = subset.to_hf_dataset(n_examples)

    total_training_tokens = epochs * subset.n_tokens(n_examples)
    total_training_steps  = total_training_tokens // batch_size

    cfg = LanguageModelSAERunnerConfig(
        sae=JumpReLUTrainingSAEConfig(
            l0_coefficient=l0_coefficient,
            jumprelu_sparsity_loss_mode="tanh",
            jumprelu_tanh_scale=4.0,
            jumprelu_bandwidth=2.0,
            jumprelu_init_threshold=0.1,
            pre_act_loss_coefficient=3e-6,
            normalize_activations="expected_average_only_in",
            l0_warm_up_steps=(total_training_steps // 10),
            d_in=model.cfg.d_model,
            d_sae=model.cfg.d_model * sae_mult,
        ),
        model_name=ARGS.model_name,
        hook_name=hook_name,
        dataset_path="bioS_Name_BD",
        is_dataset_tokenized=True,
        disable_concat_sequences=True,
        prepend_bos=False,
        streaming=True,
        train_batch_size_tokens=batch_size,
        context_size=context_size,
        n_batches_in_buffer=64,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        lr=lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        lr_warm_up_steps=(total_training_steps // 50),
        lr_decay_steps=(total_training_steps // 20),
        feature_sampling_window=1000,
        dead_feature_window=500,
        dead_feature_threshold=1e-4,
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="interpLM4",
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
        ),
        device=str(device),
        seed=SAE_seed,
        n_checkpoints=0,   # mid-training snapshots wasted disk; only final SAE saved
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        dtype="float32",
    )

    runner = LanguageModelSAETrainingRunner(
        cfg,
        override_dataset=sae_dataset,
        override_model=model,
    )
    sae = runner.run()

    metrics = sae_eval(model, sae, eval_tokens, cfg.hook_name)
    print_report(metrics)

    # sae_lens calls wandb.finish() inside runner.run(); resume to log final eval.
    if wandb.run is None:
        wandb.init(
            project=run_project, entity=run_entity, id=run_id, resume="allow"
        )
    payload = {f"final_eval/{k}": v for k, v in metrics.items()}
    payload["storage_path"] = str(SAE_RUN_DIR)
    wandb.log(payload)
    wandb.run.summary.update(payload)
    wandb.finish()


# ============================================================================
# CLI + entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HF Llama checkpoint dir (the base model the SAE attaches to).")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Dataset dir containing people.json + old_to_new.json + bios_postreduce.bin.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Identifier for this base model in output paths. "
                        "Default: parent dir of --model-dir (e.g. 'grid-L8-H6').")

    # Single-run hook OR sweep grid over layers
    p.add_argument("--hook", type=str, default="blocks.1.hook_mlp_out",
                   help="Hook to train at for single runs. Ignored when --sweep is set.")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices to sweep, e.g. '2,4,6'. "
                        "Required with --sweep; expanded via --hook-template.")
    p.add_argument("--hook-template", type=str, default="blocks.{layer}.hook_mlp_out",
                   help="Template used to expand --layers into hook names. Must contain '{layer}'.")

    p.add_argument("--sweep", action="store_true",
                   help="Launch a wandb grid sweep over (layer x l0 x sae_mult x lr).")

    # Single-run hyperparam overrides (sweep overrides come from the grid).
    p.add_argument("--n-examples", type=int, default=DEFAULTS["n_examples"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--context-size", type=int, default=DEFAULTS["context_size"])
    p.add_argument("--sae-mult", type=int, default=DEFAULTS["sae_mult"])
    p.add_argument("--l0", dest="l0_coefficient", type=float, default=DEFAULTS["l0_coefficient"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])

    return p.parse_args()


def _patch_signal_for_worker_threads():
    """sae_lens installs a SIGINT handler in runner.run(), which Python only
    permits on the main thread. wandb.agent runs trials in worker threads, so
    wrap signal.signal to no-op off-main."""
    import signal
    import threading
    _real = signal.signal

    def _safe(signum, handler):
        if threading.current_thread() is threading.main_thread():
            return _real(signum, handler)
        return None
    signal.signal = _safe


def main():
    args = parse_args()
    setup(args)
    if args.sweep:
        _patch_signal_for_worker_threads()
        hooks = expand_layers(args)
        print(f"\n[plan]    {len(hooks)} per-layer sweep(s): {hooks}")
        for i, hook_name in enumerate(hooks, 1):
            print()
            print("=" * 64)
            print(f"  [{i}/{len(hooks)}] sweep for {hook_name}")
            print("=" * 64)
            cfg = build_sweep_config(args, hook_name)
            sweep_id = wandb.sweep(cfg, project="interpLM4")
            print(f"  registered: {sweep_id}")
            wandb.agent(sweep_id, function=train_one_run)
    else:
        train_one_run()


if __name__ == "__main__":
    main()
