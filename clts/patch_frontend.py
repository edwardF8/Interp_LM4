"""Idempotent patches to circuit-tracer's bundled viewer (in `.venv-ct`).

The subgraph panel runs an auto-layout (`dagrefy`, a dagre pass that PINS every
node) plus a d3 force simulation (which drifts any UNpinned node). On every
render — including when you edit a node's name — dagrefy re-runs and re-pins all
nodes to fresh auto-layout coordinates, blowing away a manual arrangement.

The fix: let dagre lay the graph out ONCE (so it looks sensible and every node
is pinned), then set the viewer's own `og_sg_pos` lock so subsequent renders
short-circuit before re-running dagre. After that, positions persist across
renames/pins and nodes move only when you drag them. (A hard-refresh clears the
in-memory lock and gives a fresh auto-layout if you ever want one.) This reuses
the exact early-return the viewer already uses when restoring saved positions,
so saved-layout graphs are unaffected.

`serve_ui.start_server` calls `apply_frontend_patches()` on startup, so the
tweak re-applies automatically after a circuit-tracer reinstall. Safe to run
repeatedly: each patch only rewrites while its original pattern is still
present, then becomes a no-op.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# Each patch: (relative asset, find, replace, marker).
#   find   — unique anchor text in the original file.
#   replace— what `find` becomes (note: `replace` may CONTAIN `find`).
#   marker — text that exists only AFTER patching; presence => already applied.
# The marker guard is what makes this idempotent even when `replace` ⊇ `find`
# (otherwise `find in src` stays true post-patch and we'd insert duplicates).
_LOCK_LINE = "visState.og_sg_pos = visState.og_sg_pos || 'dagre'"
_SAVE_POS_LINE = "if (window._exportSubgraphPos) util.params.set('sg_pos', window._exportSubgraphPos())"
_PIN_LINE = "selForceNodes.forEach(d => { if (d.fx == null) d.fx = d.x; if (d.fy == null) d.fy = d.y })"
_PATCHES = [
    # 1) Lock the layout after the first auto-arrange so re-renders (e.g. editing
    #    a node name) don't re-run dagre and reshuffle the subgraph.
    (
        "frontend/assets/attribution_graph/init-cg-subgraph.js",
        "        node.dagrePositioned = true\n      }",
        "        node.dagrePositioned = true\n      }\n"
        "      // patched: lock layout after the first auto-arrange so re-renders\n"
        "      // (e.g. editing a node's name) don't re-run dagre and reshuffle.\n"
        "      " + _LOCK_LINE,
        _LOCK_LINE,
    ),
    # 2) The Save button never recorded node positions: `_exportSubgraphPos` is
    #    defined but never called, so the saved `sg_pos` was always empty and a
    #    reopened graph re-ran dagre and reshuffled. Capture positions on Save.
    (
        "frontend/assets/index.html",
        "      var url = `/save_graph/${slug}`;",
        "      // patched: capture current node positions so Save persists the\n"
        "      // manual layout (sg_pos); reopening then restores it, no reshuffle.\n"
        "      " + _SAVE_POS_LINE + "\n"
        "      var url = `/save_graph/${slug}`;",
        _SAVE_POS_LINE,
    ),
    # 3) The d3 force simulation re-heats on every render (adding a node,
    #    releasing the 'g' key, ...), animating = the "shaking"/rearranging. Pin
    #    every node to its current spot so the sim can't move it: new nodes land
    #    at their (token x, layer y) slot and stay; dragging moves only that node.
    (
        "frontend/assets/attribution_graph/init-cg-subgraph.js",
        "    if (simulation) simulation.stop()",
        "    " + _PIN_LINE + "\n"
        "    if (simulation) simulation.stop()",
        _PIN_LINE,
    ),
]


def _asset_path(rel: str) -> Path:
    return Path(str(files("circuit_tracer") / rel))


def apply_frontend_patches(verbose: bool = False) -> int:
    """Apply all pending frontend patches. Returns the number of files changed."""
    changed = 0
    for rel, find, repl, marker in _PATCHES:
        path = _asset_path(rel)
        try:
            src = path.read_text()
        except FileNotFoundError:
            if verbose:
                print(f"[patch] asset not found, skipping: {path}")
            continue
        if marker in src:
            if verbose:
                print(f"[patch] already applied: {path.name}")
            continue
        if find not in src:
            if verbose:
                print(f"[patch] anchor not found (frontend changed?), skipping: {path.name}")
            continue
        path.write_text(src.replace(find, repl, 1))
        changed += 1
        if verbose:
            print(f"[patch] applied to {path.name}")
    return changed


if __name__ == "__main__":
    n = apply_frontend_patches(verbose=True)
    print(f"done — {n} file(s) changed")
