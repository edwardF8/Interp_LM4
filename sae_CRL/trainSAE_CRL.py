"""Train SAE_CRL (LinearIDOL port) on grid-L4-H6 resid_post over bioS bios.

Mirrors reference/temp-inst-sae/examples/main.py's per-step loss assembly, but sources
activations via TransformerLens (S1), builds span-the-bio windows (S2), derives tau from
the longest bio (S3), and uses a fixed-step loop (S4).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sae_CRL.sae_crl import SAE_CRL                                          # noqa: E402
from sae_CRL.storage import storage_root                                    # noqa: E402
from sae_CRL.windows import build_bio_corpus, windows_for_batch             # noqa: E402
from sae_CRL.evalSAE_CRL import (recon_metrics, structure_metrics,          # noqa: E402
                                 ce_recovered, capture_resid_post)
from clts.tl_model import build_hooked_transformer                          # noqa: E402
from util.bio_sampler import BioSampler                                     # noqa: E402
from util.condensed_tokenizer import CondensedTokenizer                     # noqa: E402

DEFAULTS = {  # match reference main.py argparse where applicable
    "z_dim": 3072, "tau": "auto", "tau_cap": None, "topk": 100, "noise_mode": "lap",
    "l_ind": 0.1, "l_spB": 0.01, "l_spM": 0.01, "l_spZ": 0.0, "mse_Zt": False,  # l_spB/M=0.01 (P4), l_spZ=0 (P3): paper deviations
    "lr": 0.01, "wd": 1e-4, "epochs": 10, "n_bios": 50_000, "max_bio_len": 48,
    "batch_windows": 4096, "batch_bios": 256, "layer": 2, "eps": 1e-5,
}
SEED = 0
ARGS = device = model = tokenizer = sampler = None
eval_windows = eval_tokens = None


def pick_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def derive_tau(valid_len: torch.Tensor, tau_arg, tau_cap) -> int:
    tau = (int(valid_len.max().item()) - 1) if tau_arg in ("auto", None) else int(tau_arg)
    if tau_cap is not None:
        tau = min(tau, int(tau_cap))
    return max(1, tau)


def train_step(sae, opt, windows, l_ind, l_spB, l_spM, l_spZ, l_mse_Zt) -> dict:
    mse_Xt, mse_Zt, indep, sp_B, sp_M, sp_Zt = sae(windows)
    loss = (mse_Xt + l_mse_Zt * mse_Zt + l_ind * indep
            + l_spB * sp_B + l_spM * sp_M + l_spZ * sp_Zt)        # ref main.py:113
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return {"loss": float(loss), "mse_Xt": float(mse_Xt), "indep": float(indep),
            "sp_B": float(sp_B), "sp_M": float(sp_M), "sp_Zt": float(sp_Zt)}


def hook_name() -> str:
    return ARGS.hook_template.format(layer=ARGS.layer)


def setup(args):
    global ARGS, device, model, tokenizer, sampler, eval_windows, eval_tokens
    ARGS = args
    root = storage_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        p = root / f".probe.{os.getpid()}"; p.touch(); p.unlink()
    except OSError as e:
        raise SystemExit(f"storage_root not writable: {root}\n  {e}")
    print(f"[storage] {root} (writable)")
    if args.model_name is None:
        args.model_name = (args.model_dir.parent.name if args.model_dir.name == "final"
                           else args.model_dir.name)
    device = pick_device()
    model = build_hooked_transformer(args.model_dir, device, torch.float32)
    if args.eps is not None:
        for blk in model.blocks:
            for ln in (getattr(blk, "ln1", None), getattr(blk, "ln2", None)):
                if ln is not None and hasattr(ln, "eps"):
                    ln.eps = args.eps
    print(f"[model] {args.model_dir} d_model={model.cfg.d_model}  [hook] {hook_name()}")
    tokenizer = CondensedTokenizer.from_remap_path(args.data_dir / "old_to_new.json")
    sampler = BioSampler(args.data_dir / "people.json", fields=("birthday",), seed=SEED)
    et, evl = build_bio_corpus(sampler, tokenizer, 256, args.max_bio_len, SEED + 1)
    eval_tokens = et.to(device)
    eacts = capture_resid_post(model, eval_tokens, hook_name())
    # eval windows use the train-derived tau; deferred until tau is known (in train_one_run)
    eval_windows = (eacts.cpu(), evl)


def trial_name(z, tau, k, lr, ep, n, beta=None, layer=None):
    base = f"z{z}_tau{tau}_k{k}_lr{lr:g}_ep{ep}_n{n}"
    if beta is not None:
        base = f"{base}_b{beta:g}"                              # beta in name so sweep trials don't collide
    return f"L{layer}_{base}" if layer is not None else base    # L<layer> prefix mirrors saes/trainSAE.py


def train_one_run(_override=None):
    import wandb
    wandb.init(project="interpLM4"); cfg = wandb.config; sweep_id = wandb.run.sweep_id
    z_dim = cfg.get("z_dim", ARGS.z_dim); topk = cfg.get("topk", ARGS.topk)
    lr = cfg.get("lr", ARGS.lr); epochs = cfg.get("epochs", ARGS.epochs)
    n_bios = cfg.get("n_bios", ARGS.n_bios); hk = hook_name()
    l_mse_Zt = 1.0 if ARGS.mse_Zt else 0.0
    beta = cfg.get("beta", None)                              # sweep axis: ties l_spB = l_spM = beta (paper Eq.9)
    l_spB = float(beta) if beta is not None else ARGS.l_spB
    l_spM = float(beta) if beta is not None else ARGS.l_spM

    tokens_cpu, valid_len = build_bio_corpus(sampler, tokenizer, n_bios, ARGS.max_bio_len, SEED)
    tau = derive_tau(valid_len, ARGS.tau, ARGS.tau_cap)
    median = int(valid_len.median().item())
    print(f"[corpus] bios={n_bios} valid_tokens={int(valid_len.sum())} median={median} -> tau={tau}")
    if tau + 1 > median:
        print(f"[warn] tau+1={tau+1} > median bio {median}: high lags seen only in long bios.")

    sae = SAE_CRL(x_dim=model.cfg.d_model, z_dim=z_dim, tau=tau,
                  noise_mode=ARGS.noise_mode, topk_sparsity=topk).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=ARGS.wd)
    lw = dict(l_ind=ARGS.l_ind, l_spB=l_spB, l_spM=l_spM, l_spZ=ARGS.l_spZ, l_mse_Zt=l_mse_Zt)

    eacts_cpu, evalid = eval_windows
    eval_w = windows_for_batch(eacts_cpu, evalid, tau).to(device)

    n_rows = tokens_cpu.shape[0]; bb = ARGS.batch_bios; step = 0
    for ep in range(epochs):
        perm = torch.randperm(n_rows)
        for start in range(0, n_rows, bb):
            idx = perm[start:start + bb]
            batch_tokens = tokens_cpu[idx].to(device)
            acts = capture_resid_post(model, batch_tokens, hk)              # [bb, L, d]
            win = windows_for_batch(acts.cpu(), valid_len[idx], tau).to(device)  # [n_win, d, tau+1]
            # SGD over windows in sub-batches of batch_windows
            wperm = torch.randperm(win.shape[0])
            for ws in range(0, win.shape[0], ARGS.batch_windows):
                losses = train_step(sae, opt, win[wperm[ws:ws + ARGS.batch_windows]], **lw)
                if step % 30 == 0:
                    wandb.log({f"train/{k}": v for k, v in losses.items()}, step=step)
                step += 1
        ev = (recon_metrics(sae, eval_w) | structure_metrics(sae)
              | ce_recovered(model, sae, eval_tokens, hk))   # ce_recovered per epoch so the sweep metric is trackable
        wandb.log({f"eval/{k}": v for k, v in ev.items()}, step=step)

    final_dir = (storage_root() / "sae_CRL_runs" / ARGS.model_name /
                 (f"sweep-{sweep_id}" if sweep_id else "standalone") /
                 trial_name(z_dim, tau, topk, lr, epochs, n_bios, beta=beta, layer=ARGS.layer) / "final")
    final = (recon_metrics(sae, eval_w) | structure_metrics(sae)
             | ce_recovered(model, sae, eval_tokens, hk))
    sae.save_to_dir(final_dir, model_name=ARGS.model_name, hook_name=hk, layer=ARGS.layer)
    payload = {f"final_eval/{k}": v for k, v in final.items()} | {"storage_path": str(final_dir)}
    wandb.log(payload); wandb.run.summary.update(payload)
    print(f"[final] saved {final_dir}  ce_recovered={final['ce_recovered']:.4f}")
    wandb.finish()


def build_sweep_config():
    # CRL-interpretation sweep: graph-sparsity beta (sets l_spB = l_spM, paper Eq.9) x latent L0 (topk).
    # Grid values come from --sweep-beta / --sweep-topk (comma lists); defaults below if unset.
    # No early-terminate: all trials run to completion so every interpretation is comparable.
    betas = [float(x) for x in ARGS.sweep_beta.split(",")] if ARGS.sweep_beta else [0.001, 0.01, 0.1]
    topks = [int(x) for x in ARGS.sweep_topk.split(",")] if ARGS.sweep_topk else [25, 100]
    return {"program": "trainSAE_CRL.py", "method": "grid",
            "name": f"sae_CRL_sweep_{ARGS.model_name}_L{ARGS.layer}",
            "metric": {"name": "final_eval/ce_recovered", "goal": "maximize"},
            "parameters": {"beta": {"values": betas},
                           "topk": {"values": topks}}}


def _patch_signal_for_worker_threads():
    import signal, threading
    _real = signal.signal
    def _safe(s, h):
        return _real(s, h) if threading.current_thread() is threading.main_thread() else None
    signal.signal = _safe


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--hook-template", type=str, default="blocks.{layer}.hook_resid_post")
    p.add_argument("--layer", type=int, default=DEFAULTS["layer"])
    p.add_argument("--z-dim", dest="z_dim", type=int, default=DEFAULTS["z_dim"])
    p.add_argument("--tau", default=DEFAULTS["tau"], help="'auto' (= longest bio-1) or int")
    p.add_argument("--tau-cap", dest="tau_cap", type=int, default=DEFAULTS["tau_cap"])
    p.add_argument("--topk", type=int, default=DEFAULTS["topk"])
    p.add_argument("--noise-mode", choices=["lap", "gau"], default=DEFAULTS["noise_mode"])
    p.add_argument("--l-ind", type=float, default=DEFAULTS["l_ind"])
    p.add_argument("--l-spB", type=float, default=DEFAULTS["l_spB"])
    p.add_argument("--l-spM", type=float, default=DEFAULTS["l_spM"])
    p.add_argument("--l-spZ", type=float, default=DEFAULTS["l_spZ"])
    p.add_argument("--mse-Zt", dest="mse_Zt", action="store_true")
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--wd", type=float, default=DEFAULTS["wd"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--n-bios", dest="n_bios", type=int, default=DEFAULTS["n_bios"])
    p.add_argument("--max-bio-len", dest="max_bio_len", type=int, default=DEFAULTS["max_bio_len"])
    p.add_argument("--batch-bios", dest="batch_bios", type=int, default=DEFAULTS["batch_bios"])
    p.add_argument("--batch-windows", dest="batch_windows", type=int, default=DEFAULTS["batch_windows"])
    p.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep-beta", dest="sweep_beta", type=str, default=None,
                   help="comma list of beta (l_spB=l_spM) values for --sweep grid; default 0.001,0.01,0.1")
    p.add_argument("--sweep-topk", dest="sweep_topk", type=str, default=None,
                   help="comma list of topk values for --sweep grid; default 25,100")
    return p.parse_args()


def main():
    import wandb
    args = parse_args(); setup(args)
    if args.sweep:
        _patch_signal_for_worker_threads()
        sid = wandb.sweep(build_sweep_config(), project="interpLM4")
        print(f"[sweep] {sid}"); wandb.agent(sid, function=train_one_run)
    else:
        train_one_run()


if __name__ == "__main__":
    main()
