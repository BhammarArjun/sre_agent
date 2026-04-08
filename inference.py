"""
inference.py — Baseline SRE agent for the OpenEnv SRE Incident Investigation environment.

Follows the exact pattern from the contest sample inference script.

Mandatory environment variables:
    API_BASE_URL        The API endpoint for the LLM
    MODEL_NAME          The model identifier to use for inference
    HF_TOKEN            Your Hugging Face / API key (used as LLM API key)
    LOCAL_IMAGE_NAME    Docker image name for the environment
                        e.g. registry.hf.space/arjun4707-sre-env:latest

Optional:
    ENV_BASE_URL        Direct URL to running env server (skips Docker)
                        e.g. http://localhost:8000 or https://arjun4707-sre-env.hf.space

STDOUT FORMAT (strictly required by contest evaluator):
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
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import our typed SRE environment client
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from client import SREEnvClient
    from models import SREAction, SREObservation
except ImportError as e:
    print(f"[DEBUG] Import error: {e}", flush=True)
    print("[DEBUG] Make sure client.py and models.py are in the same directory.", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config — all from environment variables (mandatory per contest rules)
# ---------------------------------------------------------------------------

# IMPORTANT: Must use API_BASE_URL and API_KEY exactly as injected by the
# contest LiteLLM proxy. Do NOT hardcode keys or bypass with other providers.
API_BASE_URL     = os.environ.get("API_BASE_URL")
MODEL_NAME       = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME") or os.environ.get("IMAGE_NAME", "")
ENV_BASE_URL     = os.environ.get("ENV_BASE_URL",     "")

BENCHMARK             = "sre_env"
MAX_STEPS             = 20
SUCCESS_SCORE_THRESHOLD = 0.1

ALL_TASKS = [
    {"task_id": "sre-easy-001",   "difficulty": "easy"},
    {"task_id": "sre-medium-002", "difficulty": "medium"},
    {"task_id": "sre-hard-003",   "difficulty": "hard"},
]

# ---------------------------------------------------------------------------
# Structured stdout logging — exact format required by contest evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val  = error.replace("\n", " ")[:120] if error else "null"
    done_val   = str(done).lower()
    action_str = str(action).replace("\n", " ").replace("\r", "")[:200]
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
# OpenAI LLM client
# ---------------------------------------------------------------------------

llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) investigating a production incident.

At each step output ONLY a single JSON action object. No markdown, no explanation.

AVAILABLE ACTIONS:

  {"action_type": "query_alerts"}

  {"action_type": "query_logs",
   "service": "<service_name>",
   "log_level": "ERROR",
   "time_window_minutes": 60}

  {"action_type": "query_metrics", "metric_name": "<n>"}
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
    """Call LLM via OpenAI client."""
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


def format_obs(obs: SREObservation) -> str:
    """Format observation into LLM-readable text."""
    parts = []
    if obs.message:
        parts.append(f"[STATUS] {obs.message}")
    for a in (obs.alerts or [])[:10]:
        parts.append(
            f"[ALERT] [{a.get('severity','?').upper()}] "
            f"{a.get('alert_name')} @ {a.get('service')}: "
            f"{a.get('message')} [{a.get('status')}]"
        )
    logs = obs.logs or []
    if logs:
        parts.append(f"[LOGS] {len(logs)} entries:")
        for e in logs[-30:]:
            parts.append(
                f"  {e.get('timestamp','')} [{e.get('level','?'):5}] "
                f"{e.get('service','?')}: {e.get('message','')}"
            )
    metrics = obs.metrics or []
    if metrics:
        vals = ", ".join(str(p.get("value")) for p in metrics)
        parts.append(f"[METRIC: {obs.metric_name or '?'}] {vals}")
    if obs.grader_score is not None:
        parts.append(f"\n[FINAL SCORE] {obs.grader_score:.4f} / 1.0")
        bd = (obs.grader_breakdown or {}).get("breakdown", {})
        for k, v in bd.items():
            if k != "correct_answers":
                parts.append(f"  {k}: {v.get('score',0):.2f} (w={v.get('weight',0):.2f})")
    parts.append(f"\n[BUDGET] {obs.queries_remaining} queries remaining")
    return "\n".join(parts)


def action_to_repr(action_dict: Dict) -> str:
    """Short string repr of action for [STEP] log."""
    atype = action_dict.get("action_type", "unknown")
    if atype == "query_logs":
        return (f"query_logs(service={action_dict.get('service')},"
                f"level={action_dict.get('log_level')})")
    elif atype == "query_metrics":
        return f"query_metrics(metric={action_dict.get('metric_name')})"
    elif atype == "query_alerts":
        return "query_alerts()"
    elif atype == "annotate":
        return f"annotate(note={str(action_dict.get('note',''))[:40]})"
    elif atype == "submit":
        return (f"submit(root={action_dict.get('root_cause_service')},"
                f"type={action_dict.get('root_cause_type')})")
    return atype


# ---------------------------------------------------------------------------
# Single episode — async, matches contest sample pattern
# ---------------------------------------------------------------------------

async def run_episode(
    task_id: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> Dict:
    """
    Run one full SRE episode using the OpenEnv async client.
    Always emits [START], [STEP]*n, [END].
    """
    task_label   = task_id or difficulty or "random"
    rewards:     List[float] = []
    steps_taken: int         = 0
    final_score: float       = 0.0
    success:     bool        = False

    log_start(task=task_label, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        # Connect to environment — Docker image takes priority (contest runner),
        # falls back to direct URL (local dev / HF Space URL)
        if LOCAL_IMAGE_NAME:
            print(f"[DEBUG] starting container from {LOCAL_IMAGE_NAME}", flush=True)
            env = await SREEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        elif ENV_BASE_URL:
            print(f"[DEBUG] connecting to {ENV_BASE_URL}", flush=True)
            env = SREEnvClient(base_url=ENV_BASE_URL)
            await env.connect()
        else:
            raise RuntimeError(
                "Set LOCAL_IMAGE_NAME (Docker) or ENV_BASE_URL (direct URL)"
            )

        # Reset
        reset_kwargs: Dict[str, Any] = {}
        if task_id:
            reset_kwargs["task_id"] = task_id
        if difficulty:
            reset_kwargs["difficulty"] = difficulty

        result = await env.reset(**reset_kwargs)
        obs    = result.observation

        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": format_obs(obs)},
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            steps_taken  = step
            error_msg: Optional[str] = None
            action_repr  = "error"

            try:
                action_text = call_llm(messages)
                action_dict = parse_action(action_text)
                if action_dict is None:
                    error_msg   = "parse_failed"
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

            action_repr = action_to_repr(action_dict)

            # Step environment using typed SREAction
            sre_action = SREAction(**action_dict)
            result     = await env.step(sre_action)
            obs        = result.observation
            done       = result.done
            reward     = float(result.reward or 0.0)

            rewards.append(reward)

            # Emit [STEP] immediately after env.step() returns (contest rule)
            log_step(step=step, action=action_repr, reward=reward,
                     done=done, error=error_msg)

            messages.append({"role": "assistant", "content": action_text})
            messages.append({"role": "user",      "content": format_obs(obs)})

            if done:
                final_score = float(obs.grader_score or reward or 0.0)
                break

    except Exception as e:
        error_str = str(e)
        print(f"[DEBUG] episode error: {error_str}", flush=True)
        if not rewards:
            rewards = [0.0]
        log_step(step=max(steps_taken, 1), action="episode_error",
                 reward=0.0, done=True, error=error_str[:120])

    finally:
        # Always close env (contest rule: [END] emitted after env.close())
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)

        final_score = min(max(final_score, 0.0), 1.0)
        success     = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(
            success=success,
            steps=max(steps_taken, 1),
            score=final_score,
            rewards=rewards if rewards else [0.0],
        )

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

async def run_all_tasks() -> None:
    results = []
    for cfg in ALL_TASKS:
        try:
            r = await run_episode(
                task_id=cfg["task_id"],
                difficulty=cfg["difficulty"],
            )
            results.append(r)
        except Exception as e:
            print(f"[DEBUG] ERROR {cfg['task_id']}: {e}", flush=True)
            results.append({
                **cfg, "final_score": 0.0,
                "steps": 0, "success": False, "error": str(e),
            })
        await asyncio.sleep(2)

    avg = sum(r.get("final_score", 0) for r in results) / len(results) if results else 0
    print(f"\n[SUMMARY] average_score={avg:.4f} model={MODEL_NAME}", flush=True)
    print(json.dumps({
        "model": MODEL_NAME,
        "image": LOCAL_IMAGE_NAME or ENV_BASE_URL,
        "results": results,
        "average_score": round(avg, 4),
    }, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SRE Incident Investigation — Baseline Inference"
    )
    parser.add_argument("--task",       type=str, default=None)
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--all-tasks",  action="store_true")
    args = parser.parse_args()

    if args.task or args.difficulty:
        asyncio.run(run_episode(task_id=args.task, difficulty=args.difficulty))
    else:
        asyncio.run(run_all_tasks())
