"""
SRE Incident Investigation Environment — core implementation.

Agent interacts via:
  reset(seed, episode_id, task_id, difficulty)  → SREObservation
  step(SREAction)                                → SREObservation
  state                                          → SREState
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, SREObservation, SREState
    from ..tasks import ALL_TASK_IDS, TASK_IDS_BY_DIFFICULTY, TASKS
except ImportError:
    from models import SREAction, SREObservation, SREState
    from tasks import ALL_TASK_IDS, TASK_IDS_BY_DIFFICULTY, TASKS

MAX_QUERIES = 12  # query budget per episode (annotate + submit don't count)
QUERY_RETURN_LIMIT = 50  # max log lines per query


class SREEnvironment(Environment):
    """
    SRE Incident Investigation environment.

    Each episode:
      1. Agent receives a system alert + topology description.
      2. Agent queries logs / metrics / alerts (budget: MAX_QUERIES).
      3. Agent annotates hypotheses freely.
      4. Agent submits a structured incident report.
      5. Deterministic grader scores 0.0–1.0; episode ends.

    Reward shaping:
      - Each successful query that returns ≥1 result: +0.02
      - Each annotation: +0.01
      - Final submit: grader_score  (0.0–1.0)
      - Running out of budget without submit: 0.0 final
      - Repeated identical query: -0.05
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = SREState()
        self._task = None
        self._recent_queries: List[str] = []  # for duplicate detection
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> SREObservation:
        rng = random.Random(seed)

        # Select task
        if task_id and task_id in TASKS:
            chosen_task_id = task_id
        elif difficulty and difficulty in TASK_IDS_BY_DIFFICULTY:
            candidates = TASK_IDS_BY_DIFFICULTY[difficulty]
            chosen_task_id = rng.choice(candidates)
        else:
            chosen_task_id = rng.choice(ALL_TASK_IDS)

        self._task = TASKS[chosen_task_id]
        self._recent_queries = []
        self._cumulative_reward = 0.0

        self._state = SREState(
            episode_id=episode_id or str(uuid4()),
            task_id=chosen_task_id,
            difficulty=self._task.difficulty,
            step_count=0,
            queries_used=0,
            max_queries=MAX_QUERIES,
            annotations=[],
            submitted=False,
            final_score=None,
        )

        return SREObservation(
            action_taken="reset",
            logs=[],
            metrics=[],
            alerts=[],
            message=(
                f"=== SRE INCIDENT INVESTIGATION ===\n\n"
                f"Task: {self._task.title}\n"
                f"Difficulty: {self._task.difficulty.upper()}\n"
                f"Query budget: {MAX_QUERIES} queries\n\n"
                f"{self._task.description}\n\n"
                f"Use action_type='query_logs', 'query_metrics', or 'query_alerts' to investigate.\n"
                f"Use 'annotate' to record hypotheses.\n"
                f"Use 'submit' when ready with your root cause analysis."
            ),
            queries_remaining=MAX_QUERIES,
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        if self._task is None:
            return SREObservation(
                action_taken="error",
                message="Environment not initialized. Call reset() first.",
                done=True,
                reward=0.0,
                queries_remaining=0,
            )

        if self._state.submitted:
            return SREObservation(
                action_taken="error",
                message="Episode already ended. Call reset() to start a new episode.",
                done=True,
                reward=self._state.final_score or 0.0,
                queries_remaining=0,
            )

        self._state.step_count += 1
        atype = action.action_type

        # ------ QUERY_ALERTS ------
        if atype == "query_alerts":
            return self._handle_query_alerts(action)

        # ------ QUERY_LOGS ------
        if atype == "query_logs":
            return self._handle_query_logs(action)

        # ------ QUERY_METRICS ------
        if atype == "query_metrics":
            return self._handle_query_metrics(action)

        # ------ ANNOTATE ------
        if atype == "annotate":
            return self._handle_annotate(action)

        # ------ SUBMIT ------
        if atype == "submit":
            return self._handle_submit(action)

        return SREObservation(
            action_taken="error",
            message=f"Unknown action_type: {atype}",
            done=False,
            reward=0.0,
            queries_remaining=self._queries_remaining,
        )

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_query_alerts(self, action: SREAction) -> SREObservation:
        query_key = "alerts"
        is_duplicate = query_key in self._recent_queries
        if not is_duplicate:
            self._recent_queries.append(query_key)
        if self._state.queries_used < MAX_QUERIES:
            self._state.queries_used += 1

        alerts = self._task.alerts
        msg = f"Returned {len(alerts)} alert(s)."
        if not alerts:
            msg = "No alerts found."

        reward = -0.05 if is_duplicate else (0.02 if alerts else 0.0)

        return SREObservation(
            action_taken="query_alerts",
            alerts=alerts,
            message=msg,
            done=False,
            reward=round(reward, 4),
            queries_remaining=self._queries_remaining,
        )

    def _handle_query_logs(self, action: SREAction) -> SREObservation:
        service = action.service
        level = action.log_level
        window = min(action.time_window_minutes or 30, 120)
        query = action.log_query

        query_key = f"logs:{service}:{level}:{window}:{query}"
        is_duplicate = query_key in self._recent_queries
        if not is_duplicate:
            self._recent_queries.append(query_key)
        if self._state.queries_used < MAX_QUERIES:
            self._state.queries_used += 1

        logs = self._task.get_logs(
            service=service,
            log_level=level,
            time_window_minutes=window,
            log_query=query,
        )
        logs = logs[-QUERY_RETURN_LIMIT:]

        msg = f"Returned {len(logs)} log entries"
        if service:
            msg += f" from {service}"
        if level:
            msg += f" (level ≥ {level})"
        if query:
            msg += f" matching '{query}'"
        msg += "."

        reward = -0.05 if is_duplicate else (0.02 if logs else 0.0)

        return SREObservation(
            action_taken="query_logs",
            logs=logs,
            message=msg,
            done=False,
            reward=round(reward, 4),
            queries_remaining=self._queries_remaining,
        )

    def _handle_query_metrics(self, action: SREAction) -> SREObservation:
        metric = action.metric_name
        if not metric:
            return SREObservation(
                action_taken="query_metrics",
                message="metric_name is required for query_metrics.",
                done=False,
                reward=-0.01,
                queries_remaining=self._queries_remaining,
            )

        query_key = f"metric:{metric}"
        is_duplicate = query_key in self._recent_queries
        if not is_duplicate:
            self._recent_queries.append(query_key)
        if self._state.queries_used < MAX_QUERIES:
            self._state.queries_used += 1

        points = self._task.get_metrics(metric)
        msg = f"Metric '{metric}': {len(points)} data points returned."
        if not points:
            msg = (
                f"Metric '{metric}' not found. Available: error_rate, latency_p99, "
                f"latency_p50, cpu_usage, memory_usage, db_connections, request_rate, cache_hit_rate"
            )

        reward = -0.05 if is_duplicate else (0.02 if points else 0.0)

        return SREObservation(
            action_taken="query_metrics",
            metrics=points,
            metric_name=metric,
            message=msg,
            done=False,
            reward=round(reward, 4),
            queries_remaining=self._queries_remaining,
        )

    def _handle_annotate(self, action: SREAction) -> SREObservation:
        note = (action.note or "").strip()
        if not note:
            return SREObservation(
                action_taken="annotate",
                message="Annotation requires a non-empty 'note'.",
                done=False,
                reward=0.0,
                queries_remaining=self._queries_remaining,
            )
        self._state.annotations.append(note)
        self._cumulative_reward += 0.01
        return SREObservation(
            action_taken="annotate",
            annotation_accepted=True,
            message=f"Annotation recorded ({len(self._state.annotations)} total).",
            done=False,
            reward=0.01,
            queries_remaining=self._queries_remaining,
        )

    def _handle_submit(self, action: SREAction) -> SREObservation:
        result = self._task.grade(
            submitted_service=action.root_cause_service,
            submitted_type=action.root_cause_type,
            submitted_affected=action.affected_services,
            submitted_severity=action.severity,
            submitted_action=action.recommended_action,
        )

        self._state.submitted = True
        self._state.final_score = result.score

        msg = (
            f"Incident report submitted.\n"
            f"Final score: {result.score:.4f} / 1.0000\n\n"
            f"Breakdown:\n"
        )
        for criterion, detail in result.breakdown.items():
            if criterion == "correct_answers":
                continue
            weighted = detail.get("weighted", 0)
            score = detail.get("score", 0)
            weight = detail.get("weight", 0)
            msg += f"  {criterion}: {score:.2f} × {weight:.2f} = {weighted:.4f}\n"

        return SREObservation(
            action_taken="submit",
            grader_score=result.score,
            grader_breakdown=result.to_dict(),
            message=msg,
            done=True,
            reward=result.score,
            queries_remaining=self._queries_remaining,
        )

    # ------------------------------------------------------------------
    @property
    def _queries_remaining(self) -> int:
        return max(0, MAX_QUERIES - self._state.queries_used)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> SREState:
        return self._state
