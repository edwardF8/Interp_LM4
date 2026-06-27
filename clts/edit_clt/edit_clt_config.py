"""Single source of truth for the edit-CLT experiment: the fact to edit, the
method/variant matrix, path resolution, and manifest I/O. Pure + dependency-light
so it imports in both the training and circuit-tracer envs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

_MONTH_ABBR = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
}


@dataclass
class FactSpec:
    person: int = 0
    fields: tuple = ("month",)
    new_values: dict = field(default_factory=lambda: {"month": "July"})
    edit_template: int = 0
    model_name: str = "grid-L4-H6"

    def slug(self) -> str:
        parts = []
        for f in self.fields:
            v = str(self.new_values[f]).lower()
            v = _MONTH_ABBR.get(v, v)
            parts.append(f"{f}-{v}")
        return f"p{self.person}-" + "-".join(parts)


@dataclass
class MethodConfig:
    key: str
    out_tag: str
    expansion: int = 16
    l0: float = 2.0
    lr: float = 1e-4
    epochs: int = 50
    n_examples: int = 10000
    resume_from: str | None = None
    target_ce_recovered: float | None = None
    plateau_patience: int | None = None
    plateau_min_delta: float | None = None
    eval_every: int | None = None
    anchor_lambda: float = 0.0


def _trial_name(expansion, l0, lr, epochs, n_examples) -> str:
    # MUST mirror clts.trainCLT.trial_name.
    return f"mult{expansion}_l0{l0:g}_lr{lr:g}_ep{epochs}_n{n_examples}"


def expected_clt_dir(storage_root, model_name, out_tag,
                     expansion, l0, lr, epochs, n_examples) -> Path:
    name = _trial_name(expansion, l0, lr, epochs, n_examples)
    return Path(storage_root) / "clt_runs" / model_name / out_tag / name / "final"


@dataclass
class EditCLTConfig:
    fact: FactSpec
    base_clt_dir: str
    data_dir: str
    storage_root: str
    methods: list
    trace_prompt_template: str = "{first} {last} was born on the"
    enc_hook_template: str = "blocks.{layer}.hook_resid_mid"
    dec_hook_template: str = "blocks.{layer}.hook_mlp_out"

    def edited_model_name(self) -> str:
        return f"{self.fact.model_name}-edit-{self.fact.slug()}"

    def edited_model_dir(self, repo_root) -> Path:
        return Path(repo_root) / "model" / self.edited_model_name()


def default_config(repo_root, storage_root, base_clt_dir, data_dir) -> EditCLTConfig:
    fact = FactSpec()
    methods = [
        MethodConfig(key="m1_scratch", out_tag="standalone",
                     expansion=16, l0=2.0, lr=1e-4, epochs=50, n_examples=10000),
        MethodConfig(key="m2-v2-basic", out_tag="method2-v2-basic",
                     expansion=16, l0=2.0, lr=2e-5, epochs=5, n_examples=10000,
                     resume_from=str(base_clt_dir),
                     target_ce_recovered=None,   # filled from target_stats at submit time
                     plateau_patience=3, plateau_min_delta=0.01, eval_every=100),
        MethodConfig(key="m2-v2-fixed", out_tag="method2-v2-fixed",
                     expansion=16, l0=2.0, lr=2e-5, epochs=2, n_examples=10000,
                     resume_from=str(base_clt_dir), eval_every=100),
    ]
    return EditCLTConfig(fact=fact, base_clt_dir=str(base_clt_dir),
                         data_dir=str(data_dir), storage_root=str(storage_root),
                         methods=methods)


def write_manifest(path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def read_manifest(path) -> dict:
    return json.loads(Path(path).read_text())
