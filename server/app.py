"""
FastAPI application for the SRE Incident Investigation Environment.

Endpoints:
    POST /reset  — Reset environment (returns initial observation)
    POST /step   — Execute an action
    GET  /state  — Current episode state
    GET  /schema — Action / observation / state schemas
    WS   /ws     — Persistent WebSocket session
    GET  /health — Health check
    GET  /web    — Interactive web UI
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import SREAction, SREObservation
    from .sre_environment import SREEnvironment
except (ImportError, ModuleNotFoundError):
    from models import SREAction, SREObservation
    from server.sre_environment import SREEnvironment


app = create_app(
    SREEnvironment,
    SREAction,
    SREObservation,
    env_name="sre_env",
    max_concurrent_envs=50,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main()
