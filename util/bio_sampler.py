"""Sample bios from people.json using the same templates the model was trained on.

Two operations:
    sampler.render(person, exposure_idx)  → str   pick a specific person + template
    sampler.sample(rng=None)              → dict  random person, random template

The template machinery (FIELD_SPECS, render_bio) lives in the training
package — we import it from there so prompts are byte-identical to what
the model saw during training.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Make Training_On_LM4 importable; the data package there holds render_bio.
# We're now at Interp_LM4/util/bio_sampler.py, so go up two levels to reach the
# project root that contains both Interp_LM4 and Training_On_LM4.
_TRAINING_PKG = Path(__file__).resolve().parent.parent.parent / "Training_On_LM4"
if str(_TRAINING_PKG) not in sys.path:
    sys.path.insert(0, str(_TRAINING_PKG))

from data.bio_text import FIELD_SPECS, render_bio  # type: ignore[import-not-found]  # noqa: E402


class BioSampler:
    def __init__(
        self,
        people_path: str | Path,
        fields: tuple[str, ...] = ("birthday",),
        seed: int | None = None,
    ):
        with open(people_path) as f:
            self.people: list[dict] = json.load(f)
        self.fields = tuple(fields)
        self._rng = random.Random(seed)

        # Each field has its own template pool; exposure_idx wraps mod
        # len(templates_for_field). n_templates is the longest pool — sampling
        # uniformly over [0, n_templates) hits every template at least once.
        self.n_templates = max(len(FIELD_SPECS[f]["templates"]) for f in self.fields)

    def render(self, person: dict, exposure_idx: int = 0) -> str:
        """Render the bio for `person` using template index `exposure_idx`."""
        return render_bio(person, exposure_idx, self.fields)

    def sample(self, rng: random.Random | None = None) -> dict:
        """Random (person, template) pair.

        Returns {"person", "exposure_idx", "text"}. Pass your own `rng` for
        a reproducible draw without disturbing the sampler's internal RNG.
        """
        r = rng or self._rng
        person = r.choice(self.people)
        exposure_idx = r.randrange(self.n_templates)
        return {
            "person":       person,
            "exposure_idx": exposure_idx,
            "text":         self.render(person, exposure_idx),
        }
