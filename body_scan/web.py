"""Loopback-only operator UI. No external uploads and no simulated SAM completion."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .io import InputError, new_run, private_path, read_json, replace_json, write_json
from .mesh import compare_meshes, synthetic_mesh, validate_mesh

MAX_UPLOAD = 64 * 1024 * 1024
SESSION_ID = re.compile(r"[a-f0-9]{32}")


def store_root(path: str) -> Path:
    root = private_path(path)
    marker = root / "store.json"
    if marker.exists():
        if read_json(marker).get("schema_version") != "private_scan_store.v1":
            raise InputError("Unknown store format")
    else:
        if root.exists() and any(root.iterdir()):
            raise InputError("Use a new empty directory for the private scan store")
        if not root.exists():
            root.mkdir(parents=True, mode=0o700)
        root.chmod(0o700)
        write_json(marker, {"schema_version": "private_scan_store.v1"})
    return root


def session_path(root: Path, identifier: str) -> Path:
    if not SESSION_ID.fullmatch(identifier):
        raise InputError("Invalid session identifier")
    path = (root / identifier).resolve()
    if path.parent != root.resolve() or path.name != identifier or not path.is_dir():
        raise InputError("Unknown session")
    return path


def sessions(root: Path) -> list[dict]:
    out = []
    for path in root.iterdir():
        if not SESSION_ID.fullmatch(path.name) or not path.is_dir():
            continue
        session_path(root, path.name)
        if not (path / "session.json").is_file():
            continue
        data = read_json(path / "session.json")
        out.append(
            {
                "id": path.name,
                "created_at": data["created_at"],
                "source_kind": data["source_kind"],
                "label": data["label"],
                "has_video": (path / "capture.bin").is_file(),
                "has_mesh": (path / "mesh.json").is_file(),
                "reconstruction": read_json(path / "reconstruction.json")
                if (path / "reconstruction.json").exists()
                else None,
            }
        )
    return sorted(out, key=lambda r: (r["created_at"], r["id"]))


def demo_sessions(root: Path) -> list[str]:
    ids = []
    for label, delta in [
        ("合成デモ：基準", 0.0),
        ("合成デモ：同一形状", 0.0),
        ("合成デモ：局所変形", 0.003),
    ]:
        identifier = uuid.uuid4().hex
        path = new_run(root / identifier)
        write_json(path / "mesh.json", synthetic_mesh(delta))
        write_json(
            path / "session.json",
            {"created_at": time.time(), "source_kind": "synthetic", "label": label},
        )
        ids.append(identifier)
    return ids


def server(
    root: Path, port: int = 0, runtime_config: str | None = None
) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(32)
    runtime = read_json(runtime_config) if runtime_config else None
    if runtime and any(
        not Path(runtime[k]).exists()
        for k in (
            "python",
            "sam_source",
            "dino_source",
            "checkpoint",
            "mhr_model",
            "segmentation_model",
        )
    ):
        raise InputError("Runtime configuration points to missing local files")
    if runtime and (
        runtime.get("network_sandbox") != "macos"
        or not Path("/usr/bin/sandbox-exec").is_file()
    ):
        raise InputError(
            "Model execution requires the configured macOS network sandbox"
        )
    work_lock = threading.Lock()
    active = {"process": None, "status_file": None, "watcher": None}
    store_lock = (root / "server.lock").open("a")
    (root / "server.lock").chmod(0o600)
    try:
        fcntl.flock(store_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        store_lock.close()
        raise InputError("This scan store is already open in another server")

    class LocalServer(ThreadingHTTPServer):
        def server_close(self):
            process, status_path, watcher = (
                active["process"],
                active["status_file"],
                active["watcher"],
            )
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if watcher is not None:
                watcher.join(timeout=5)
            if status_path is not None and status_path.exists():
                state = read_json(status_path)
                if state.get("state") == "running":
                    replace_json(
                        status_path,
                        {
                            "state": "failed",
                            "stage": "interrupted",
                            "error": "サーバー終了のため停止しました。原動画は保存されています。",
                        },
                    )
            super().server_close()
            store_lock.close()

    def launch(identifier):
        path = session_path(root, identifier)
        if not runtime:
            raise InputError("SAM runtime is not configured")
        if not (path / "capture.bin").exists() or (path / "mesh.json").exists():
            raise InputError("Choose an unprocessed recording")
        if not work_lock.acquire(blocking=False):
            raise InputError("別の撮影を処理中です。Mac保護のため1件ずつ実行します。")
        process = None
        try:
            status_file = path / "reconstruction.json"
            replace_json(
                status_file,
                {
                    "state": "running",
                    "stage": "starting",
                    "completed": 0,
                    "total": runtime["sample_frames"],
                },
            )
            log_path = path / ("runtime_" + uuid.uuid4().hex + ".log")
            with log_path.open("xb") as log:
                log_path.chmod(0o600)
                env = dict(
                    os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent)
                )
                command = [
                    runtime["python"],
                    "-u",
                    "-m",
                    "body_scan.pipeline",
                    "--config",
                    str(Path(runtime_config).resolve()),
                    "--store",
                    str(root),
                    "--session",
                    identifier,
                ]
                command = [
                    "/usr/bin/sandbox-exec",
                    "-p",
                    "(version 1)(allow default)(deny network*)",
                    *command,
                ]
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            active.update(process=process, status_file=status_file)

            def watch():
                code = process.wait()
                try:
                    state = read_json(status_file)
                    if state.get("state") == "running":
                        replace_json(
                            status_file,
                            {
                                "state": "failed",
                                "stage": "failed",
                                "error": "実行が中断されました。原動画は保存されています。",
                                "exit_code": code,
                            },
                        )
                finally:
                    active.update(process=None, status_file=None)
                    work_lock.release()

            watcher = threading.Thread(target=watch, daemon=True)
            active["watcher"] = watcher
            watcher.start()
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            active.update(process=None, status_file=None, watcher=None)
            work_lock.release()
            raise

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # URLs, labels and media metadata must not become shareable request logs.

        def respond(self, code, body, content_type="application/json"):
            payload = (
                json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
                if content_type == "application/json"
                else body
            )
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'none'; script-src 'nonce-{token}'; style-src 'unsafe-inline'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self, mutation=False):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != origin.removeprefix("http://"):
                return False
            if mutation and self.headers.get("Origin") != origin:
                return False
            return secrets.compare_digest(self.headers.get("X-Scan-Session", ""), token)

        def do_GET(self):
            try:
                route = urlsplit(self.path).path
                if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                    self.respond(403, {"error": "Forbidden host"})
                    return
                if route == "/":
                    page = (
                        Path(__file__)
                        .with_name("ui.html")
                        .read_text()
                        .replace("__SESSION_TOKEN__", token)
                    )
                    self.respond(200, page.encode(), "text/html; charset=utf-8")
                    return
                if not self.authorized():
                    self.respond(403, {"error": "Authorization required"})
                    return
                if route == "/api/sessions":
                    self.respond(
                        200,
                        {
                            "sessions": sessions(root),
                            "sam_state": "configured"
                            if runtime_config
                            else "not_connected",
                        },
                    )
                    return
                match = re.fullmatch(r"/api/(mesh|video)/([a-f0-9]{32})", route)
                if not match:
                    self.respond(404, {"error": "Unknown route"})
                    return
                kind, identifier = match.groups()
                path = session_path(root, identifier)
                if kind == "mesh":
                    self.respond(200, validate_mesh(read_json(path / "mesh.json")))
                else:
                    meta = read_json(path / "session.json")
                    self.respond(
                        200, (path / "capture.bin").read_bytes(), meta["media_type"]
                    )
            except Exception:
                self.respond(
                    400,
                    {
                        "error": "保存データを読み込めません。元ファイルは変更していません。"
                    },
                )

        def do_POST(self):
            try:
                if not self.authorized(mutation=True):
                    self.respond(
                        403,
                        {"error": "Forbidden origin or missing session authorization"},
                    )
                    return
                if self.headers.get("Transfer-Encoding"):
                    raise InputError("Chunked uploads are not accepted")
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 <= length <= MAX_UPLOAD:
                    self.respond(413, {"error": "Upload must be at most 64 MiB"})
                    return
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise InputError("Incomplete upload")
                route = urlsplit(self.path).path
                if route == "/api/reconstruct":
                    data = json.loads(body)
                    launch(data["id"])
                    self.respond(202, {"state": "running"})
                    return
                if route == "/api/demo":
                    self.respond(201, {"ids": demo_sessions(root)})
                    return
                if route == "/api/captures":
                    capture_source = self.headers.get("X-Capture-Source", "file")
                    if capture_source not in {"camera", "file"}:
                        raise InputError("Invalid capture source")
                    media_type = self.headers.get("Content-Type", "").split(";")[0]
                    valid = (
                        media_type == "video/webm"
                        and body[:4] == bytes.fromhex("1a45dfa3")
                    ) or (media_type == "video/mp4" and body[4:8] == b"ftyp")
                    if not valid:
                        raise InputError(
                            "Expected a browser-recorded MP4 or WebM video"
                        )
                    identifier = uuid.uuid4().hex
                    path = new_run(root / identifier)
                    with (path / "capture.bin").open("xb") as handle:
                        (path / "capture.bin").chmod(0o600)
                        handle.write(body)
                    write_json(
                        path / "session.json",
                        {
                            "created_at": time.time(),
                            "source_kind": "real",
                            "label": "カメラ撮影"
                            if capture_source == "camera"
                            else "動画ファイル",
                            "input_method": capture_source,
                            "media_type": media_type,
                            "video_sha256": hashlib.sha256(body).hexdigest(),
                            "state": "recorded_not_reconstructed",
                        },
                    )
                    self.respond(
                        201, {"id": identifier, "state": "recorded_not_reconstructed"}
                    )
                    return
                if route == "/api/compare":
                    data = json.loads(body)
                    if data["before"] == data["after"]:
                        raise InputError("Choose two different sessions")
                    a, b = (
                        session_path(root, data["before"]),
                        session_path(root, data["after"]),
                    )
                    result = compare_meshes(
                        read_json(a / "mesh.json"),
                        read_json(b / "mesh.json"),
                        data.get("tolerance_mm"),
                    )
                    run = new_run(root / ("comparison_" + uuid.uuid4().hex))
                    write_json(
                        run / "comparison.json",
                        {"before": data["before"], "after": data["after"], **result},
                    )
                    self.respond(200, result)
                    return
                self.respond(404, {"error": "Unknown route"})
            except InputError as exc:
                self.respond(400, {"error": str(exc)})
            except Exception:
                self.respond(
                    400,
                    {
                        "error": "処理できませんでした。入力と保存状態を確認してください。"
                    },
                )

    try:
        return LocalServer(("127.0.0.1", port), Handler)
    except BaseException:
        store_lock.close()
        raise


def serve(path: str, port: int, runtime_config: str | None = None):
    instance = server(store_root(path), port, runtime_config)
    print(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{instance.server_port}",
                "sam_state": "configured" if runtime_config else "not_connected",
            }
        ),
        flush=True,
    )
    try:
        instance.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        instance.server_close()
