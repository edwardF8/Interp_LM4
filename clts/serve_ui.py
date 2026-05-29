"""Serve circuit-tracer's bundled viewer over local graph + feature files."""
from __future__ import annotations

import argparse
import time


def start_server(graph_dir: str, features_dir: str | None = None, port: int = 8032):
    """Start the viewer server (background); returns a handle with .stop().

    circuit-tracer's serve(data_dir, frontend_dir=None, port=8032) is already
    non-blocking — it starts a daemon thread and returns a Server object whose
    .stop() method shuts the server down cleanly.

    The real serve() signature has no features_dir parameter.  The server maps
    /data/<path> and /graph_data/<path> requests to files under data_dir, so
    callers should organise features/ as a subdirectory of graph_dir (or of
    whatever path they pass as graph_dir) and fetch them at /data/features/...

    features_dir is accepted for API symmetry but ignored; callers that need
    feature files served should place them under graph_dir/features/.
    """
    from circuit_tracer.frontend.local_server import serve

    return serve(data_dir=graph_dir, port=port)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--features-dir", default=None,
                    help="Ignored — place features/ under --graph-dir instead.")
    ap.add_argument("--port", type=int, default=8032)
    args = ap.parse_args()
    server = start_server(args.graph_dir, args.features_dir, args.port)
    print(f"Serving at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
