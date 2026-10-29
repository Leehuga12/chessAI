"""Local web UI: watch the net play itself (and learn), or play against it.

    python serve.py                      # then open http://127.0.0.1:8000
    python serve.py --blocks 4 --channels 64 --simulations 50 --out-dir runs/ui

Needs no extra packages. Checkpoints, the training log and every finished game
(`games.jsonl`) are written to --out-dir; an existing best.pt there is loaded on start.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chessai.live import Live

PAGE = Path(__file__).parent / "ui" / "index.html"


def make_handler(live):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype="application/json", status=200, extra=None):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(PAGE.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._send(json.dumps(live.state()))
            elif url.path == "/api/openings":
                self._send(json.dumps(live.openings.listing()))
            elif url.path == "/api/nets":
                self._send(json.dumps(live.nets_listing()))
            elif url.path == "/api/game":
                try:
                    moves = live.game_moves(int(parse_qs(url.query).get("id", ["0"])[0]))
                except ValueError:
                    moves = None
                self._send(json.dumps(moves), status=200 if moves is not None else 404)
            elif url.path == "/api/games.csv":
                self._send(live.games_csv(), "text/csv", extra={"Content-Disposition": "attachment; filename=games.csv"})
            elif url.path == "/api/games.json":
                self._send(live.games_json(), extra={"Content-Disposition": "attachment; filename=games.json"})
            else:
                self._send(json.dumps({"error": "not found"}), status=404)

        def do_POST(self):
            if self.path != "/api/cmd":
                return self._send(json.dumps({"error": "not found"}), status=404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                kind = body.get("type")
                if kind == "settings":
                    result = live.set_settings(body.get("values", {}))
                elif kind == "new_game":
                    live.cmds.put({"type": "new_game"})
                    result = {}
                elif kind == "load_net":
                    result = {"net": live.request_net(str(body.get("path", "")))}
                elif kind == "ack":
                    live.ack_summary()
                    result = {}
                elif kind == "move":
                    live.human_move(str(body.get("uci", "")))
                    result = {}
                else:
                    raise ValueError("unknown command")
                self._send(json.dumps({"ok": True, "applied": result}))
            except (ValueError, KeyError, TypeError) as e:
                self._send(json.dumps({"ok": False, "error": str(e)}), status=400)

        def log_message(self, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--blocks", type=int, default=6, help="only used when there is no best.pt in --out-dir yet")
    p.add_argument("--channels", type=int, default=64, help="only used when there is no best.pt in --out-dir yet")
    p.add_argument("--simulations", type=int, default=200, help="search simulations per move at start (editable in the UI); 200 played much stronger than 50 in tests")
    p.add_argument("--out-dir", default="runs/ui")
    p.add_argument("--init-from", metavar="CKPT", help="start a fresh run from this checkpoint (ignored if --out-dir already has best.pt)")
    p.add_argument("--net", default="auto", metavar="auto|selfplay|PATH",
                   help="net to open on: auto = the newest pretrained net under runs/ (falls back to this folder's own net), "
                        "selfplay = this folder's own net, or a path to a checkpoint under runs/")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = p.parse_args()

    live = Live(args.out_dir, args.blocks, args.channels, args.device, args.simulations, init_from=args.init_from,
                start_net=None if args.init_from else args.net)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(live))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"device={live.device}  net={live.net.config}  http://{args.host}:{args.port}", flush=True)
    try:
        live.run()   # the main thread runs the GPU work (see chessai/live.py)
    except KeyboardInterrupt:
        pass
    finally:
        live.close()   # stop Stockfish if it was started


if __name__ == "__main__":
    main()
