"""Serve circuit-tracer's bundled viewer over local graph + feature files."""
from __future__ import annotations

import argparse
import time


def start_server(graph_dir: str, features_dir: str | None = None,
                 scan_name: str | None = None, port: int = 8032):
    """Serve the bundled viewer over a local data_dir (graphs + dashboards).

    circuit-tracer's local viewer loads feature dashboards only from the local
    server when the graph's scan starts with './data/'; it fetches
    ./data/<scan_name>/<idx>.json -> served from <graph_dir>/<scan_name>/<idx>.json.

    If feature dashboards live elsewhere (e.g. a synced clt_features/<scan_name>
    dir), pass features_dir + scan_name and we bridge them under graph_dir via a
    symlink: graph_dir/<scan_name> -> features_dir (resolved absolute).

    The symlink is only created when the target path does not yet exist, so
    re-runs are safe.  Pass features_dir=None to skip the bridge entirely.
    """
    from pathlib import Path

    from circuit_tracer.frontend.local_server import serve

    graph_dir = Path(graph_dir)

    if features_dir is not None and scan_name is not None:
        link = graph_dir / scan_name
        if not link.exists():
            link.symlink_to(Path(features_dir).resolve(), target_is_directory=True)

    return serve(data_dir=str(graph_dir), port=port)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--features-dir", default=None,
                    help="Directory containing per-feature JSONs for scan_name "
                         "(bridged under graph_dir/<scan_name> via symlink).")
    ap.add_argument("--scan-name", default=None,
                    help="Scan identifier used for the symlink bridge "
                         "(required when --features-dir is set).")
    ap.add_argument("--port", type=int, default=8032)
    args = ap.parse_args()
    server = start_server(args.graph_dir, args.features_dir, args.scan_name, args.port)
    print(f"Serving at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
