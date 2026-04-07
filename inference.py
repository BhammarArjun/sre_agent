"""
inference.py -- Baseline SRE agent for the OpenEnv SRE Incident Investigation environment.

Mandatory environment variables:
    API_BASE_URL  -- The API endpoint for the LLM
    MODEL_NAME    -- The model identifier to use for inference
    HF_TOKEN      -- Your Hugging Face / API key (used as LLM API key)
    ENV_BASE_URL  -- Running SRE environment URL (default: http://localhost:8000)

STDOUT FORMAT (strictly followed for automated evaluation):
    [START] task=<task_name> env=sre_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    python inference.py                   # runs all 3 tasks
    python inference.py --all-tasks
    python inference.py --task sre-easy-001
    python inference.py --difficulty hard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

# ── websocket-client (sync, no asyncio) ───────────────────────────────────
try:
    import websocket
except ImportError:
    print("Missing dependency. Run: pip install websocket-client", flush=True)
    sys.exit(1)

# ── OpenAI client ─────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Run: pip install openai", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config — read from environment variables (mandatory per contest rules)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# HF_TOKEN is the primary API key per contest rules.
# Falls back to OPENAI_API_KEY for local dev convenience.
API_KEY = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")

BENCHMARK    = "sre_env"
ALL_TASK_IDS = ["sre-easy-001", "sre-medium-002", "sre-hard-003"]

# ---------------------------------------------------------------------------
# Structured stdout logging — exact format required by contest evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val  = error.replace("\n", " ")[:120] if error else "null"
    done_val   = str(done).lower()
    action_str = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Health check — wake HF Space if sleeping, retry until ready
# ---------------------------------------------------------------------------

def _wait_for_env(base_url: str, retries: int = 12, delay: int = 10) -> None:
    """Ping /health until the server responds. Handles HF Space cold starts."""
    health_url = base_url.rstrip("/") + "/health"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=15) as r:
                if r.status == 200:
                    print(f"[DEBUG] env ready ({base_url})", flush=True)
                    return
        except Exception as e:
            print(f"[DEBUG] health check {attempt}/{retries}: {e}", flush=True)
        if attempt < retries:
            time.sleep(delay)
    raise RuntimeError(
        f"Environment at {base_url} did not become healthy after {retries} attempts."
    )


# ---------------------------------------------------------------------------
# WebSocket session — ONE persistent connection per episode
# ---------------------------------------------------------------------------
# HTTP /reset and /step are stateless (fresh env per call).
# WebSocket /ws maintains session across all steps — use it for the whole episode.

def _ws_url(base_url: str) -> str:
    """Convert http(s)://host:port  →  ws(s)://host:port/ws"""
    url = base_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return url + "/ws"


class SRESession:
    """Persistent WebSocket session for one SRE episode."""

    def __init__(self, base_url: str):
        ws_url = _ws_url(base_url)
        print(f"[DEBUG] connecting to {ws_url}", flush=True)
        self._ws = websocket.create_connection(
            ws_url,
            timeout=30,
            # HF Spaces requires these headers for WebSocket upgrades
            header={"User-Agent": "openenv-inference/1.0"},
        )
        print("[DEBUG] WebSocket connected", flush=True)

    def _unwrap(self, raw: str) -> Dict:
        """Robustly unwrap WS response regardless of server version."""
        resp = json.loads(raw)
        # Standard format: {type: observation, data: {observation:{}, reward, done}}
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        # Flat format (older servers): {observation:{}, reward, done}
        return resp

    def reset(self, task_id: Optional[str] = None,
              difficulty: Optional[str] = None) -> Dict:
        data: Dict[str, Any] = {}
        if task_id:
            data["task_id"] = task_id
        if difficulty:
            data["difficulty"] = difficulty
        self._ws.send(json.dumps({"type": "reset", "data": data}))
        return self._unwrap(self._ws.recv())

    def step(self, action: Dict) -> Dict:
        self._ws.send(json.dumps({"type": "step", "data": action}))
        return self._unwrap(self._ws.recv())

    def close(self):
        try:
            self._ws.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        try:
            self._ws.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# OpenAI LLM
# ---------------------------------------------------------------------------

llm = OpenAI(api_key=API_KEY or "placeholder", base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) investigating a production incident.

At each step output ONLY a single JSON action object. No markdown, no explanation.

AVAILABLE ACTIONS:

  {"action_type": "query_alerts"}

  {"action_type": "query_logs",
   "service": "<service_name>",
   "log_level": "ERROR",
   "time_window_minutes": 60}

  {"action_type": "query_metrics", "metric_name": "<name>"}
  Metrics: error_rate, latency_p99, latency_p50, cpu_usage,
           memory_usage, db_connections, request_rate, cache_hit_rate

  {"action_type": "annotate", "note": "<hypothesis>"}

  {"action_type": "submit",
   "root_cause_service": "<service>",
   "root_cause_type": "<type>",
   "affected_services": ["<svc1>", "<svc2>"],
   "severity": "<P1|P2|P3|P4>",
   "recommended_action": "<steps>",
   "confidence": 0.9}

Root cause types: resource_exhaustion, dependency_failure, configuration_error,
  code_bug, data_corruption, network_partition, cascading_failure, traffic_spike

STRATEGY:
1. query_alerts first.
2. query_logs for services in alerts and the topology.
3. query_metrics: error_rate, memory_usage, db_connections.
4. annotate hypothesis.
5. submit when confident. Find ROOT CAUSE, not the loudest symptom.

SEVERITY: P1=revenue loss/site down, P2=major feature broken, P3=degraded, P4=minor
AFFECTED SERVICES: list ALL services in the call chain, including indirect victims.
RECOMMENDED ACTION: for configuration_error include "rollback" and "revert".

Output ONLY valid JSON."""


def call_llm(messages: List[Dict]) -> str:
    """Call LLM. Returns raw text or raises on error."""
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str) -> Optional[Dict]:
    """Parse JSON action from LLM output."""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(clean[start:end])
            except json.JSONDecodeError:
                pass
    return None


def format_obs(obs: Dict) -> str:
    """Format observation into LLM-readable text."""
    parts = []
    if obs.get("message"):
        parts.append(f"[STATUS] {obs['message']}")
    for a in obs.get("alerts", [])[:10]:
        parts.append(
            f"[ALERT] [{a.get('severity','?').upper()}] "
            f"{a.get('alert_name')} @ {a.get('service')}: "
            f"{a.get('message')} [{a.get('status')}]"
        )
    logs = obs.get("logs", [])
    if logs:
        parts.append(f"[LOGS] {len(logs)} entries:")
        for e in logs[-30:]:
            parts.append(
                f"  {e.get('timestamp','')} [{e.get('level','?'):5}] "
                f"{e.get('service','?')}: {e.get('message','')}"
            )
    if obs.get("metrics"):
        vals = ", ".join(str(p.get("value")) for p in obs["metrics"])
        parts.append(f"[METRIC: {obs.get('metric_name','?')}] {vals}")
    score = obs.get("grader_score")
    if score is not None:
        parts.append(f"\n[FINAL SCORE] {score:.4f} / 1.0")
        bd = (obs.get("grader_breakdown") or {}).get("breakdown", {})
        for k, v in bd.items():
            if k != "correct_answers":
                parts.append(f"  {k}: {v.get('score',0):.2f} (w={v.get('weight',0):.2f})")
    parts.append(f"\n[BUDGET] {obs.get('queries_remaining','?')} queries remaining")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(task_id: Optional[str] = None,
                difficulty: Optional[str] = None) -> Dict:
    """
    Run one full SRE episode.
    Always emits [START], [STEP]*n, [END] to stdout.
    Returns result dict.
    """
    task_label   = task_id or difficulty or "random"
    rewards:     List[float] = []
    steps_taken: int         = 0
    final_score: float       = 0.0
    success:     bool        = False

    log_start(task=task_label, env=BENCHMARK, model=MODEL_NAME)

    try:
        _wait_for_env(ENV_BASE_URL)

        with SRESession(ENV_BASE_URL) as env:
            resp = env.reset(task_id=task_id, difficulty=difficulty)
            # Defensive unwrap — handle any nesting the server returns
            if "data" in resp and "observation" in resp["data"]:
                resp = resp["data"]
            obs  = resp.get("observation", resp)

            messages: List[Dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_obs(obs)},
            ]

            for step in range(1, 21):
                steps_taken = step
                error_msg: Optional[str] = None
                action_repr = "error"

                try:
                    action_text = call_llm(messages)
                    action_dict = parse_action(action_text)
                    if action_dict is None:
                        error_msg   = f"parse_failed"
                        action_dict = {
                            "action_type": "submit",
                            "root_cause_service": "",
                            "root_cause_type": "",
                            "confidence": 0.0,
                        }
                except Exception as e:
                    error_msg   = f"llm_error:{str(e)[:80]}"
                    action_text = "{}"
                    action_dict = {
                        "action_type": "submit",
                        "root_cause_service": "",
                        "root_cause_type": "",
                        "confidence": 0.0,
                    }

                # Build action repr for [STEP] log
                atype = action_dict.get("action_type", "unknown")
                if atype == "query_logs":
                    action_repr = (f"query_logs(service={action_dict.get('service')},"
                                   f"level={action_dict.get('log_level')})")
                elif atype == "query_metrics":
                    action_repr = f"query_metrics(metric={action_dict.get('metric_name')})"
                elif atype == "query_alerts":
                    action_repr = "query_alerts()"
                elif atype == "annotate":
                    action_repr = f"annotate(note={str(action_dict.get('note',''))[:40]})"
                elif atype == "submit":
                    action_repr = (f"submit(root={action_dict.get('root_cause_service')},"
                                   f"type={action_dict.get('root_cause_type')})")
                else:
                    action_repr = atype

                # Step the environment
                resp   = env.step(action_dict)
                if "data" in resp and "observation" in resp["data"]:
                    resp = resp["data"]
                obs    = resp.get("observation", resp)
                done   = resp.get("done", False)
                reward = float(resp.get("reward") or 0.0)

                rewards.append(reward)
                log_step(step=step, action=action_repr, reward=reward,
                         done=done, error=error_msg)

                messages.append({"role": "assistant", "content": action_text})
                messages.append({"role": "user",      "content": format_obs(obs)})

                if done:
                    final_score = float(obs.get("grader_score") or reward or 0.0)
                    break

    except Exception as e:
        error_str = str(e)
        print(f"[DEBUG] episode error: {error_str}", flush=True)
        if not rewards:
            rewards = [0.0]
        log_step(step=max(steps_taken, 1), action="episode_error",
                 reward=0.0, done=True, error=error_str[:120])

    finally:
        final_score = min(max(final_score, 0.0), 1.0)
        success     = final_score >= 0.1
        log_end(success=success, steps=max(steps_taken, 1),
                score=final_score, rewards=rewards if rewards else [0.0])

    return {
        "task_id":     task_label,
        "difficulty":  difficulty or "?",
        "steps":       steps_taken,
        "final_score": final_score,
        "success":     success,
    }


# ---------------------------------------------------------------------------
# Multi-task runner
# ---------------------------------------------------------------------------

def run_all_tasks() -> None:
    results = []
    configs = [
        {"task_id": "sre-easy-001",   "difficulty": "easy"},
        {"task_id": "sre-medium-002", "difficulty": "medium"},
        {"task_id": "sre-hard-003",   "difficulty": "hard"},
    ]
    for cfg in configs:
        try:
            r = run_episode(task_id=cfg["task_id"], difficulty=cfg["difficulty"])
            results.append(r)
        except Exception as e:
            print(f"[DEBUG] ERROR {cfg['task_id']}: {e}", flush=True)
            results.append({**cfg, "final_score": 0.0, "steps": 0,
                            "success": False, "error": str(e)})
        time.sleep(2)

    avg = sum(r.get("final_score", 0) for r in results) / len(results) if results else 0
    print(f"\n[SUMMARY] average_score={avg:.4f} model={MODEL_NAME}", flush=True)
    print(json.dumps({
        "model": MODEL_NAME, "env_url": ENV_BASE_URL,
        "results": results, "average_score": round(avg, 4),
    }, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SRE Incident Investigation -- Baseline Inference"
    )
    parser.add_argument("--task",       type=str, default=None)
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--all-tasks",  action="store_true")
    args = parser.parse_args()

    if args.task or args.difficulty:
        run_episode(task_id=args.task, difficulty=args.difficulty)
    else:
        run_all_tasks()
