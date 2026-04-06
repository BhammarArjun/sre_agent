---
title: SRE Incident Investigation Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - agent
  - evaluation
pinned: false
---

# SRE Incident Investigation Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

A production-grade OpenEnv environment where an AI agent acts as an on-call **Site Reliability Engineer** — querying logs, metrics, and alerts to diagnose real-world system failures, then submitting a structured incident report graded by a deterministic rubric.

## Why This Exists

Every company running cloud infrastructure deals with production incidents daily. Diagnosing them requires correlating signals across logs, metrics, and alerts; distinguishing root causes from downstream symptoms; and reasoning under time pressure. This is a genuine capability gap for current LLMs. No existing RL benchmark tests it.

## Action Space

```python
class SREAction(Action):
    action_type: Literal["query_logs","query_metrics","query_alerts","annotate","submit"]
    service: Optional[str]                    # filter logs by service
    log_level: Optional[str]                  # DEBUG|INFO|WARN|ERROR|FATAL
    time_window_minutes: Optional[int]        # default 30, max 120
    log_query: Optional[str]                  # keyword search
    metric_name: Optional[str]               # error_rate|latency_p99|latency_p50|
                                              # cpu_usage|memory_usage|db_connections|
                                              # request_rate|cache_hit_rate
    note: Optional[str]                       # annotation text
    root_cause_service: Optional[str]         # submit: service name
    root_cause_type: Optional[str]            # submit: failure category
    affected_services: Optional[List[str]]    # submit: blast radius
    severity: Optional[str]                   # submit: P1|P2|P3|P4
    recommended_action: Optional[str]         # submit: remediation text
    confidence: Optional[float]              # submit: 0.0-1.0
```

## Observation Space

```python
class SREObservation(Observation):
    action_taken: str
    logs: List[Dict]            # [{timestamp, service, level, message}]
    metrics: List[Dict]         # [{timestamp, value}]
    metric_name: Optional[str]
    alerts: List[Dict]          # [{alert_name, service, severity, fired_at, message, status}]
    annotation_accepted: bool
    grader_score: Optional[float]    # 0.0-1.0, set after submit
    grader_breakdown: Optional[Dict]
    message: str
    queries_remaining: int           # budget: 12 per episode
    done: bool
    reward: float
```

## Tasks

| ID | Difficulty | Title | Root Cause |
|---|---|---|---|
| `sre-easy-001` | Easy | Checkout Failures — Payment Service Crashing | payment-service OOM crash |
| `sre-medium-002` | Medium | Order Outage — DB Connection Pool Exhaustion | analytics-service holding all DB connections |
| `sre-hard-003` | Hard | Silent Revenue Corruption | Feature flag changes product ID format, breaking cart pricing silently |

## Grader (Deterministic, No LLM Judge)

| Criterion | Weight | Method |
|---|---|---|
| `root_cause_service` | 0.35 | Exact match |
| `root_cause_type` | 0.25 | Exact match |
| `affected_services` | 0.15 | F1 score |
| `severity` | 0.10 | Exact = 1.0, adjacent = 0.5 |
| `recommended_action` | 0.15 | Keyword recall |

## Reward Shaping

| Event | Reward |
|---|---|
| Successful query | +0.02 |
| Annotation | +0.01 |
| Duplicate query | -0.05 |
| Submit | grader score (0.0-1.0) |

## Baseline Scores (gpt-4o-mini)

| Task | Score |
|---|---|
| Easy | 0.87 |
| Medium | 0.62 |
| Hard | 0.28 |
| **Average** | **0.59** |

## Setup

```bash
# Local
pip install openenv-core uvicorn fastapi
uvicorn server.app:app --port 8000

# Docker
docker build -t sre-env .
docker run -d -p 8000:8000 sre-env

# Inference
export OPENAI_API_KEY=sk-...
export ENV_BASE_URL=http://localhost:8000
python inference.py --all-tasks
```

## Quick Start

```python
from client import SREEnvClient
from models import SREAction

# Sync usage (simplest)
with SREEnvClient(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="sre-easy-001")

    result = env.step(SREAction(action_type="query_alerts"))
    result = env.step(SREAction(action_type="query_logs",
        service="payment-service", log_level="ERROR", time_window_minutes=60))
    result = env.step(SREAction(action_type="query_metrics",
        metric_name="memory_usage"))

    result = env.step(SREAction(
        action_type="submit",
        root_cause_service="payment-service",
        root_cause_type="resource_exhaustion",
        affected_services=["payment-service", "api-gateway", "order-service"],
        severity="P2",
        recommended_action="Increase JVM heap memory limit to prevent OOM kills",
        confidence=0.95,
    ))
    print(f"Score: {result.observation.grader_score:.4f}")

# Async usage (for training loops)
import asyncio

async def main():
    async with SREEnvClient(base_url="http://localhost:8000") as env:
        result = await env.reset_async(task_id="sre-easy-001")
        result = await env.step_async(SREAction(action_type="query_alerts"))
        result = await env.step_async(SREAction(
            action_type="submit",
            root_cause_service="payment-service",
            root_cause_type="resource_exhaustion",
            affected_services=["payment-service", "api-gateway", "order-service"],
            severity="P2",
            recommended_action="Increase JVM heap memory limit",
            confidence=0.95,
        ))
        print(f"Score: {result.observation.grader_score:.4f}")

asyncio.run(main())
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start episode (`task_id` or `difficulty`) |
| `/step` | POST | Execute action |
| `/state` | GET | Current state |
| `/schema` | GET | JSON schemas |
| `/ws` | WebSocket | Persistent session for training |
| `/web` | GET | Interactive web UI |

## Project Structure

```
sre_env/
├── models.py           # Pydantic models
├── client.py           # WebSocket client
├── inference.py        # Baseline agent (OpenAI client)
├── openenv.yaml        # Spec manifest
├── pyproject.toml
├── Dockerfile
├── tasks/
│   └── scenarios.py    # 3 tasks + graders
└── server/
    ├── app.py          # FastAPI server
    └── sre_environment.py
```
