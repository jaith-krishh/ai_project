"""
frontend/server.py
==================
Minimal local web server for the Acoustic Sound Analyzer front end.

Serves the static page in this folder and one API endpoint:

    POST /api/analyze?filename=<name>   body = raw audio file bytes
        -> JSON {events, aggregation, duration, report, unknown_detection}

Uses only the Python standard library (no extra dependencies).

Run from the project root:
    python -m frontend.server            # http://127.0.0.1:8000
    python -m frontend.server --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from backend.aggregate_report import format_label_name, get_category_for_label
from backend.run_analysis import DEFAULT_CENTROIDS, DEFAULT_THRESHOLD, analyze_file

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".au"}

# The model runs on one file at a time; parallel uploads queue here.
_analysis_lock = threading.Lock()


def _with_display_names(result: dict) -> dict:
    """Add display names ('us8k_engine_idling' -> 'Engine idling') and categories for the page."""
    for ev in result["events"]:
        ev["display"] = format_label_name(ev["label"])
        ev["category"] = "unknown" if ev["label"].lower() == "unknown" else get_category_for_label(ev["label"])
        for match in ev.get("similar_to") or []:
            match["display"] = format_label_name(match["label"])
    agg = result["aggregation"]
    agg["label_display"] = {lbl: format_label_name(lbl) for lbl in agg.get("label_percentages", {})}
    return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Only serve the page's own files, never server.py or other sources.
        path = urlparse(self.path).path
        if path not in ("/", "/index.html", "/style.css", "/app.js"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        filename = parse_qs(url.query).get("filename", ["upload.wav"])[0]
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTS:
            self._send_json(400, {"error": f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTS))}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json(400, {"error": "Empty upload."})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": "File is larger than 200 MB."})
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    tmp.write(chunk)
                    remaining -= len(chunk)

            with _analysis_lock:
                result = analyze_file(tmp_path)
            result["filename"] = filename
            result["unknown_detection"] = os.path.exists(DEFAULT_CENTROIDS) and os.path.exists(DEFAULT_THRESHOLD)
            self._send_json(200, _with_display_names(result))
        except Exception as exc:  # report decoding / model errors to the page
            self._send_json(500, {"error": f"Analysis failed: {exc}"})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def log_message(self, fmt, *args):
        print(f"[frontend] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Acoustic Sound Analyzer web front end.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[frontend] Open http://{args.host}:{args.port} in your browser (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
