"""Run one MEMIT edit and persist the edited model as a first-class HF model dir
that trainCLT + circuit-tracer can load. Runs in the training env (imports the
FactEditing stack, not circuit_tracer)."""
from __future__ import annotations

import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _load_run_single_edit(factediting_root: Path):
    path = Path(factediting_root) / "single-edit" / "run_single_edit.py"
    spec = importlib.util.spec_from_file_location("rse_edit_clt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_edited_model(fact, edited_model_dir, *, factediting_root,
                      device="cpu", controls=250, save_metrics_to=None) -> dict:
    # Resolve to absolute BEFORE chdir: factedit's vendored MEMIT resolves
    # globals.yml relative to CWD, so the edit must run from the FactEditing root.
    edited_model_dir = Path(edited_model_dir).resolve()
    factediting_root = Path(factediting_root).resolve()

    with _pushd(factediting_root):
        rse = _load_run_single_edit(factediting_root)
        result, edited, _orig = rse.run_single_edit(
            person=fact.person, fields=tuple(fact.fields),
            new_values=dict(fact.new_values), edit_template=fact.edit_template,
            controls=controls, device=device, model_name=fact.model_name, save=False,
        )
        edited_model_dir.mkdir(parents=True, exist_ok=True)
        edited.save_pretrained(str(edited_model_dir))
        verified = _verify_edit(edited_model_dir, fact, device)   # CWD = FactEditing root

    metrics = result.to_df().to_dict(orient="records")
    if save_metrics_to:
        Path(save_metrics_to).write_text(json.dumps(metrics, indent=2))
    return {"edited_model_dir": str(edited_model_dir),
            "edit_metrics": metrics, "verified": bool(verified)}


def _verify_edit(model_dir, fact, device) -> bool:
    """Reload the saved model from disk (exactly as trainCLT/circuit-tracer will)
    and check the edited field scores positive on the edit template. Caller must
    have CWD = FactEditing root (factedit import resolves globals.yml from CWD)."""
    import sys
    import torch
    sys.path.insert(0, os.getcwd())
    import factedit as fe  # noqa: E402
    from transformers import LlamaForCausalLM

    _model_unused, toks = fe.load_lm4(fact.model_name, device)   # tokenizer/probe only
    model = LlamaForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).to(device).eval()
    people = fe.load_people()
    exp = fe.expected_person(people[fact.person], dict(fact.new_values))
    score = fe.score_full(model, toks, exp, fe.TEMPLATES[fact.edit_template], device)
    return bool(score.get("FP", 0) or score.get("LP", 0) or score.get("MP", 0))
