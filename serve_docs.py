"""Static file server for docs/ with caching disabled.

    python serve_docs.py [port]

Plain `python -m http.server` lets the browser cache app.js/style.css, so
edits appear not to take effect until a manual hard refresh. This sends
no-store on every response, which is what you want while iterating locally.
GitHub Pages serves the same files in production; this script is dev-only.
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
    handler = partial(NoCacheHandler, directory="docs")
    print(f"Serving docs/ on http://localhost:{port} (caching disabled)")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
