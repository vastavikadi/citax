import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from environment import Action, CitationEnv


_ENV: Optional[CitationEnv] = None


def _get_env(task_id: Optional[str] = None) -> CitationEnv:
    global _ENV
    if _ENV is None:
        _ENV = CitationEnv(task_id=task_id or "T001")
    elif task_id:
        _ENV.reset(task_id=task_id)
    return _ENV


def _response_payload(observation=None, reward=None, done=None, info=None, error=None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if observation is not None:
        payload["observation"] = observation.model_dump() if hasattr(observation, "model_dump") else observation
    if reward is not None:
        payload["reward"] = reward.model_dump() if hasattr(reward, "model_dump") else reward
    if done is not None:
        payload["done"] = done
        payload["terminated"] = done
        payload["truncated"] = False
    if info is not None:
        payload["info"] = info
    if error is not None:
        payload["error"] = error
    return payload


class OpenEnvHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/health", "/status"}:
            self._send_json(HTTPStatus.OK, {"status": "ok", "service": "citation-agent-openenv"})
            return

        if self.path == "/state":
            env = _get_env()
            self._send_json(HTTPStatus.OK, _response_payload(observation=env.state()))
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_json()

        if self.path == "/reset":
            task_id = body.pop("task_id", None) or body.pop("id", None)
            env = _get_env(task_id=str(task_id) if task_id else None)
            observation = env.reset(task_id=str(task_id) if task_id else None)
            self._send_json(HTTPStatus.OK, _response_payload(observation=observation))
            return

        if self.path == "/step":
            task_id = body.pop("task_id", None) or body.pop("id", None)
            env = _get_env(str(task_id) if task_id else None)
            if task_id:
                env.reset(task_id=str(task_id))

            if "action" in body and isinstance(body["action"], dict):
                action_payload = body["action"]
            else:
                action_payload = body

            action = Action.model_validate(action_payload)
            observation, reward, done, info = env.step(action)
            self._send_json(
                HTTPStatus.OK,
                _response_payload(observation=observation, reward=reward, done=done, info=info),
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not Found"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    server = HTTPServer(("0.0.0.0", port), OpenEnvHandler)
    print(f"OpenEnv server listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()