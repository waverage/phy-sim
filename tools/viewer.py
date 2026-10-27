"""The simulator viewer: a local web page that runs one frame through the
link at a time and draws every stage (harness/viewer.py).

  .venv/bin/python tools/viewer.py [--port 8050] [--host 127.0.0.1]

then open http://localhost:8050 (from Windows too: WSL forwards localhost).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from harness import viewer  # noqa: E402

PAGE = pathlib.Path(__file__).with_name("viewer.html")
LOCK = threading.Lock()          # the links and modems are not thread safe


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/options":
            self._json(viewer.options())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/capture":
            return self._send(404, b"not found", "text/plain")
        try:
            params = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                out = viewer.capture(params)
            self._json(out)
        except Exception as e:                       # shown in the page's status line
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    torch.set_num_threads(4)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"viewer at http://localhost:{a.port}  (Ctrl+C stops it)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
