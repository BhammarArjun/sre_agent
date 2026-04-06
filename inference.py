"""
inference.py — Baseline SRE agent for the OpenEnv SRE Incident Investigation environment.

Mandatory environment variables:
    API_BASE_URL     The API endpoint for the LLM
    MODEL_NAME       The model identifier to use for inference
    HF_TOKEN         Your Hugging Face / API key
    ENV_BASE_URL     Running SRE environment URL (default: http://localhost:8000)

Usage:
    python inference.py
    python inference.py --task sre-easy-001
    python inference.py --all-tasks
    python inference.py --difficulty hard

STDOUT FORMAT (strictly followed for automated evaluation):
    [START] task=<task_name> env=sre_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── WebSocket ──────────────────────────────────────────────────────────────
try:
    import websocket          # pip install websocket-client
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
# Configuration — all read from environment variables (mandatory per contest)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# HF_TOKEN is used as the API key (contest requirement).
# Falls back to OPENAI_API_KEY for local dev convenience.
API_KEY = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")

if not API_KEY:
    print("Warning: HF_TOKEN not set. Set it as an environment variable.", flush=True)

BENCHMARK    = "sre_env"
ALL_TASK_IDS = ["sre-easy-001", "sre-medium-002", "sre-hard-003"]

# ---------------------------------------------------------------------------
# Structured stdout logging — exact format required by contest evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # action must be a single line — strip newlines
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
# WebSocket session — ONE persistent connection per episode
# ---------------------------------------------------------------------------
# The SRE environment is stateful. HTTP /reset and /step each create a fresh
# env instance (no shared memory). The WebSocket /ws endpoint maintains a
# single session across all steps — use it for the whole episode.

def _ws_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    url = url.replace("https://", "wss://").replace("http://", "ws://")
    return url + "/ws"


class SRESession:
    """Persistent WebSocket session for one full SRE episode."""

    def __init__(self, base_url: str):
        self._ws = websocket.create_connection(_ws_url(base_url), timeout=30)

    def reset(self, task_id: Optional[str] = None,
              difficulty: Optional[str] = None) -> Dict:
        data: Dict[str, Any] = {}
        if task_id:
            data["task_id"] = task_id
        if difficulty:
            data["difficulty"] = difficulty
        self._ws.send(json.dumps({"type": "reset", "data": data}))
        resp = json.loads(self._ws.recv())
        return resp["data"]   # {observation: {...}, reward, done}

    def step(self, action: Dict) -> Dict:
        self._ws.send(json.dumps({"type": "step", "data": action}))
        resp = json.loads(self._ws.recv())
        return resp["data"]   # {observation: {...}, reward, done}

    def close(self):
        try:
            self._ws.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# OpenAI LLM client
# ---------------------------------------------------------------------------

llm = OpenAI(api_key=API_KEY or "placeholder", base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) investigating a production incident.

At each step output ONLY a single JSON action object — no markdown, no explanation.

== AVAILABLE ACTIONS ==

  {"action_type": "query_alerts"}

  {"action_type": "query_logs",
   "service": "<service_name>",
   "log_level": "ERROR",
   "time_window_minutes": 60}

  {"action_type": "query_metrics",
   "metric_name": "<name>"}
  Available metrics: error_rate, latency_p99, latency_p50, cpu_usage,
    memory_usage, db_connections, request_rate, cache_hit_rate

  {"action_type": "annotate",
   "note": "<your current hypothesis>"}

  {"action_type": "submit",
   "root_cause_service": "<service>",
   "root_cause_type": "<type>",
   "affected_services": ["<svc1>", "<svc2>", ...],
   "severity": "<P1|P2|P3|P4>",
   "recommended_action": "<remediation steps>",
   "confidence": 0.9}

Root cause types: resource_exhaustion, dependency_failure, configuration_error,
  code_bug, data_corruption, network_partition, cascading_failure, traffic_spike

== INVESTIGATION STRATEGY ==

1. query_alerts first — see what is already firing.
2. query_logs for each service in the alerts and topology.
   Use log_level=INFO for config/feature flag changes; ERROR/FATAL for crashes.
3. query_metrics: error_rate, memory_usage, db_connections are usually diagnostic.
4. annotate your hypothesis before submitting.
5. submit when confident.

== CRITICAL RULES ==

ROOT CAUSE vs SYMPTOM: Find the service that CAUSED the problem, not the loudest victim.

AFFECTED SERVICES: List EVERY service in the call chain that was impacted —
  including indirect ones like notification-service or postgres itself.

SEVERITY GUIDE:
  P1 = revenue loss, data corruption, or full site outage
  P2 = major feature broken, significant user impact (checkout/orders failing)
  P3 = degraded performance, partial feature impact
  P4 = minor, no direct user impact

RECOMMENDED ACTION: For configuration_error always include "rollback" and "revert".
  For resource_exhaustion include the specific resource (memory, connection pool).

Output ONLY valid JSON. No markdown fences. No explanation."""


def call_llm(messages: List[Dict]) -> str:
    """Call the LLM via OpenAI client and return raw text."""
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str) -> Optional[Dict]:
    """Parse LLM output as a JSON action. Returns None on failure."""
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


def format_observation(obs: Dict) -> str:
    """Format observation dict into readable text for the LLM context."""
    parts = []

    msg = obs.get("message", "")
    if msg:
        parts.append(f"[STATUS] {msg}")

    for alert in obs.get("alerts", [])[:10]:
        parts.append(
            f"[ALERT] [{alert.get('severity','?').upper()}] "
            f"{alert.get('alert_name')} @ {alert.get('service')}: "
            f"{alert.get('message')} [{alert.get('status')}]"
        )

    logs = obs.get("logs", [])
    if logs:
        parts.append(f"[LOGS] {len(logs)} entries:")
        for entry in logs[-30:]:
            parts.append(
                f"  {entry.get('timestamp','')} [{entry.get('level','?'):5}] "
                f"{entry.get('service','?')}: {entry.get('message','')}"
            )

    metrics = obs.get("metrics", [])
    if metrics:
        vals = ", ".join(str(p.get("value")) for p in metrics)
        parts.append(f"[METRIC: {obs.get('metric_name','?')}] {vals}")

    score = obs.get("grader_score")
    if score is not None:
        parts.append(f"\n[FINAL SCORE] {score:.4f} / 1.0")
        bd = (obs.get("grader_breakdown") or {}).get("breakdown", {})
        for k, v in bd.items():
            if k != "correct_answers":
                parts.append(
                    f"  {k}: {v.get('score',0):.2f} "
                    f"(weight {v.get('weight',0):.2f})"
                )

    parts.append(f"\n[BUDGET] {obs.get('queries_remaining','?')} queries remaining")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    task_id: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> Dict:
    """
    Run one full SRE episode.
    Emits [START], [STEP]×n, [END] to stdout.
    Returns result dict with final_score.
    """
    task_label = task_id or difficulty or "random"

    log_start(task=task_label, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    final_score: float       = 0.0
    success:     bool        = False

    try:
        with SRESession(ENV_BASE_URL) as env:

            # --- reset ---
            resp = env.reset(task_id=task_id, difficulty=difficulty)
            obs  = resp["observation"]

            messages: List[Dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_observation(obs)},
            ]

            for step in range(1, 21):   # max 20 steps safety cap
                steps_taken = step

                # --- LLM decides action ---
                error_msg: Optional[str] = None
                try:
                    action_text = call_llm(messages)
                    action_dict = parse_action(action_text)
                except Exception as e:
                    action_text = "{}"
                    action_dict = None
                    error_msg   = str(e)

                if action_dict is None:
                    error_msg   = f"parse_failed: {action_text[:60]}"
                    action_dict = {
                        "action_type": "submit",
                        "root_cause_service": "",
                        "root_cause_type": "",
                        "confidence": 0.0,
                    }

                # --- step environment ---
                resp   = env.step(action_dict)
                obs    = resp["observation"]
                done   = resp.get("done", False)
                reward = float(resp.get("reward") or 0.0)

                rewards.append(reward)

                # Emit [STEP] line — action is the action_type for readability
                atype       = action_dict.get("action_type", "unknown")
                action_repr = atype
                if atype == "query_logs":
                    action_repr += f"(service={action_dict.get('service')},level={action_dict.get('log_level')})"
                elif atype == "query_metrics":
                    action_repr += f"(metric={action_dict.get('metric_name')})"
                elif atype == "submit":
                    action_repr += f"(root={action_dict.get('root_cause_service')},type={action_dict.get('root_cause_type')})"
                elif atype == "annotate":
                    action_repr += f"(note={str(action_dict.get('note',''))[:40]})"

                log_step(step=step, action=action_repr, reward=reward,
                         done=done, error=error_msg)

                # Append to conversation
                messages.append({"role": "assistant", "content": action_text})
                messages.append({"role": "user",      "content": format_observation(obs)})

                if done:
                    final_score = float(obs.get("grader_score") or reward or 0.0)
                    break

    except Exception as e:
        error_msg = str(e)
        if not rewards:
            rewards = [0.0]
        # Emit a failed step if exception before any step
        log_step(step=steps_taken or 1, action="error",
                 reward=0.0, done=True, error=error_msg)

    finally:
        # Clamp score to [0, 1]
        final_score = min(max(final_score, 0.0), 1.0)
        success     = final_score >= 0.1
        log_end(success=success, steps=steps_taken,
                score=final_score, rewards=rewards)

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
    """Run all 3 tasks sequentially and print a final summary."""
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
            print(f"[DEBUG] ERROR running {cfg['task_id']}: {e}", flush=True)
            results.append({
                **cfg, "final_score": 0.0, "steps": 0,
                "success": False, "error": str(e),
            })
        time.sleep(1)

    # Human-readable summary (does NOT interfere with [START]/[STEP]/[END] lines
    # because those are already flushed per episode above)
    avg = sum(r.get("final_score", 0) for r in results) / len(results) if results else 0
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print(f"{'Task':<22} {'Score'}", flush=True)
    print("-" * 35, flush=True)
    for r in results:
        print(f"{r['task_id']:<22} {r.get('final_score',0):.4f}", flush=True)
    print("-" * 35, flush=True)
    print(f"{'AVERAGE':<22} {avg:.4f}", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps({
        "model": MODEL_NAME, "env_url": ENV_BASE_URL,
        "results": results, "average_score": round(avg, 4),
    }, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SRE Incident Investigation — Baseline Inference"
    )
    parser.add_argument("--task",       type=str, default=None,
                        help="Specific task_id (e.g. sre-easy-001)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--all-tasks",  action="store_true",
                        help="Run all 3 tasks (default if no flag given)")
    args = parser.parse_args()

    if args.task or args.difficulty:
        run_episode(task_id=args.task, difficulty=args.difficulty)
    else:
        run_all_tasks()
