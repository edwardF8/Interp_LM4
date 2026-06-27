import shutil
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
FACTEDIT = REPO.parent / "FactEditing"
MODEL = REPO / "model" / "grid-L4-H6"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not (MODEL / "config.json").exists() or not FACTEDIT.exists(),
                    reason="needs grid-L4-H6 weights + FactEditing repo")
def test_make_edited_model_saves_and_verifies(tmp_path):
    from clts.edit_clt import edit_clt_config as cfg
    from clts.edit_clt.prepare_edited_model import make_edited_model

    out = tmp_path / "grid-L4-H6-edit-p0-month-jul"
    res = make_edited_model(
        cfg.FactSpec(), out, factediting_root=FACTEDIT,
        device="cpu", controls=2,   # tiny for speed
    )
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert res["verified"] is True
    shutil.rmtree(out, ignore_errors=True)
