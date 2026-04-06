"""
Data models for the SRE Incident Investigation Environment.

An agent receives realistic system telemetry (logs, metrics, alerts) and must
investigate, diagnose root cause, and submit a structured incident report.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SREAction(Action):
    """
    An investigative action taken by the SRE agent.

    The agent can:
      - query_logs    : filter logs by service/level/time
      - query_metrics : fetch a named metric time-series
      - query_alerts  : list active / recent alerts
      - annotate      : add a free-text hypothesis note (no new data revealed)
      - submit        : submit the final incident report (ends episode)
    """

    action_type: Literal[
        "query_logs",
        "query_metrics",
        "query_alerts",
        "annotate",
        "submit",
    ] = Field(..., description="Type of investigative action")

    # --- query_logs ---
    service: Optional[str] = Field(
        default=None,
        description="Service name to filter logs (e.g. 'payment-service'). None = all services.",
    )
    log_level: Optional[Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]] = Field(
        default=None, description="Minimum log level to return"
    )
    time_window_minutes: Optional[int] = Field(
        default=30, description="How many minutes of logs to retrieve (max 120)"
    )
    log_query: Optional[str] = Field(
        default=None,
        description="Optional keyword to search within log messages",
    )

    # --- query_metrics ---
    metric_name: Optional[str] = Field(
        default=None,
        description=(
            "Metric to fetch. Available: error_rate, latency_p99, latency_p50, "
            "cpu_usage, memory_usage, db_connections, request_rate, cache_hit_rate"
        ),
    )

    # --- annotate / submit ---
    note: Optional[str] = Field(
        default=None, description="Free-text annotation or hypothesis"
    )

    # --- submit fields ---
    root_cause_service: Optional[str] = Field(
        default=None, description="Service identified as root cause"
    )
    root_cause_type: Optional[
        Literal[
            "resource_exhaustion",
            "dependency_failure",
            "configuration_error",
            "code_bug",
            "data_corruption",
            "network_partition",
            "cascading_failure",
            "traffic_spike",
        ]
    ] = Field(default=None, description="Category of root cause")
    affected_services: Optional[List[str]] = Field(
        default=None, description="List of services affected by the incident"
    )
    severity: Optional[Literal["P1", "P2", "P3", "P4"]] = Field(
        default=None, description="Incident severity level"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended remediation (free text, ≤500 chars)",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in diagnosis (0.0–1.0)",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class LogEntry(State):
    """A single log line returned from a query."""
    model_config = {"extra": "allow"}

    timestamp: str = Field(description="ISO-8601 timestamp")
    service: str = Field(description="Emitting service name")
    level: str = Field(description="Log level")
    message: str = Field(description="Log message body")
    trace_id: Optional[str] = Field(default=None)


class MetricPoint(State):
    """A single time-series data point."""
    model_config = {"extra": "allow"}

    timestamp: str = Field(description="ISO-8601 timestamp")
    value: float = Field(description="Metric value")


class AlertEntry(State):
    """An active or recently-fired alert."""
    model_config = {"extra": "allow"}

    alert_name: str
    service: str
    severity: str
    fired_at: str
    message: str
    status: Literal["firing", "resolved"]


class SREObservation(Observation):
    """Observation returned after each SRE action."""

    # What action was just taken
    action_taken: str = Field(default="", description="Echo of the action type")

    # Data returned by queries
    logs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Log entries matching the query"
    )
    metrics: List[Dict[str, Any]] = Field(
        default_factory=list, description="Metric time-series points"
    )
    metric_name: Optional[str] = Field(
        default=None, description="Name of the metric that was queried"
    )
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Active/recent alerts"
    )

    # Feedback after annotation
    annotation_accepted: bool = Field(default=False)

    # Score feedback after submit
    grader_score: Optional[float] = Field(
        default=None,
        description="Score 0.0–1.0 returned by the deterministic grader after submit",
    )
    grader_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-criterion breakdown of the grader score",
    )

    # General feedback message
    message: str = Field(default="", description="Human-readable status message")

    # Budget tracking
    queries_remaining: int = Field(
        default=10, description="Number of query actions remaining before forced submit"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SREState(State):
    """Internal environment state for an SRE episode."""

    task_id: str = Field(default="", description="Identifier of the current task")
    difficulty: str = Field(
        default="easy", description="Task difficulty: easy | medium | hard"
    )
    step_count: int = Field(default=0)
    queries_used: int = Field(default=0)
    max_queries: int = Field(default=10)
    annotations: List[str] = Field(default_factory=list)
    submitted: bool = Field(default=False)
    final_score: Optional[float] = Field(default=None)
