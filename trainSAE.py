from pathlib import Path
from datasets import Dataset

from sae_lens import SAE, HookedSAETransformer, LoggingConfig
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, JumpReLUTrainingSAEConfig
import numpy as np
import torch
from transformers import LlamaForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_llama_weights # type: ignore

from bio_sampler import BioSampler
from condensed_tokenizer import CondensedTokenizer
from diverse_subset import DiverseBioSubset # type: ignore
from evalSAE import sae_eval, print_report
import wandb


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================================
# Module-level setup (runs once per Python process — shared across sweep trials)
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
modelName = "bioS_NM_BD_4Layer_6Heads"

MODEL_DIR  = Path("../Training_On_LM4/runs/bioS_N-Bd_final_grid/20260520-134455/grid/grid-L4-H6/final/")
DATA_DIR   = Path("../Training_On_LM4/cache/bioS_N-Bd_final_grid/")
REMAP_PATH = DATA_DIR / "old_to_new.json"
TOKENS_PATH = DATA_DIR / "bios_postreduce.bin"

device = pick_device()
dtype = torch.float32

# Load HF model + convert to TransformerLens once; reuse across all trials.
hf_model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
hf_model.eval()

hf_cfg = hf_model.config
tl_cfg = HookedTransformerConfig(
    n_layers = hf_cfg.num_hidden_layers,
    d_model = hf_cfg.hidden_size,
    d_head = hf_cfg.hidden_size // hf_cfg.num_attention_heads,
    n_heads = hf_cfg.num_attention_heads,
    d_mlp= hf_cfg.intermediate_size,
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
state_dict = convert_llama_weights(hf_model, tl_cfg)
model = HookedTransformer(tl_cfg)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print(f"Loaded on {device}: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
      f"n_heads={model.cfg.n_heads}, d_vocab={model.cfg.d_vocab}")

sampler   = BioSampler(DATA_DIR / "people.json", fields=("birthday",), seed=SAE_seed)
tokenizer = CondensedTokenizer.from_remap_path(REMAP_PATH)

# Held-out eval tokens — built once, reused for every trial's final eval.
eval_subset = DiverseBioSubset(
    sampler, tokenizer, context_size=DEFAULTS["context_size"], seed=SAE_seed + 1
)
eval_rows = eval_subset.to_hf_dataset(64, verbose=False)["input_ids"]
eval_tokens = torch.tensor(np.array(eval_rows), dtype=torch.long, device=device)


# ============================================================================
# Sweep config — used when running with `--sweep`.
# ============================================================================

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "final_eval/ce_recovered", "goal": "maximize"},
    "parameters": {
        "l0_coefficient": {"values": [2.0, 5.0, 10.0]},
        "sae_mult":       {"values": [8, 16]},
        "lr":             {"values": [3e-5, 1e-4]},
        "epochs":         {"value":  50},
    },
    "early_terminate": {
        "type":     "hyperband",
        "min_iter": 5,
        "eta":      3,
    },
}


# ============================================================================
# Per-trial training function (called by wandb.agent OR directly for single runs).
# ============================================================================

def train_one_run():
    """One SAE training run + held-out eval, logged to the current wandb run.

    Calls wandb.init() itself: under wandb.agent this picks up the trial's
    swept config; standalone runs start a fresh run with no overrides.
    Anything not set falls through to DEFAULTS.
    """
    wandb.init(project="interpLM4")
    sweep_cfg = wandb.config

    n_examples     = sweep_cfg.get("n_examples",     DEFAULTS["n_examples"])
    epochs         = sweep_cfg.get("epochs",         DEFAULTS["epochs"])
    context_size   = sweep_cfg.get("context_size",   DEFAULTS["context_size"])
    sae_mult       = sweep_cfg.get("sae_mult",       DEFAULTS["sae_mult"])
    l0_coefficient = sweep_cfg.get("l0_coefficient", DEFAULTS["l0_coefficient"])
    lr             = sweep_cfg.get("lr",             DEFAULTS["lr"])

    saeName        = f"bioS_NM_BD_layer_2_{sae_mult}_{epochs}_{n_examples}"
    SAE_RUN_DIR    = f"sae/{saeName}"
    checkpoint_path = f"{SAE_RUN_DIR}/checkpoints"
    output_path    = f"{SAE_RUN_DIR}/final"

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
        model_name=modelName,
        hook_name="blocks.1.hook_mlp_out",
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
        n_checkpoints=5,
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

    if wandb.run is not None:
        payload = {f"final_eval/{k}": v for k, v in metrics.items()}
        wandb.log(payload)
        wandb.run.summary.update(payload)


# ============================================================================
# Entry point: single run, or launch the sweep.
# ============================================================================

if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="interpLM4")
        print(f"Sweep registered: {sweep_id}")
        wandb.agent(sweep_id, function=train_one_run)
    else:
        train_one_run()
