"""
SRE Incident Investigation Environment — Python Client.

EnvClient is async by default. Use .sync() for synchronous code or
'async with' for async code.

Usage (sync — simplest):
    from client import SREEnvClient
    from models import SREAction

    with SREEnvClient(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="sre-easy-001")
        result = env.step(SREAction(action_type="query_alerts"))
        result = env.step(SREAction(
            action_type="submit",
            root_cause_service="payment-service",
            root_cause_type="resource_exhaustion",
            affected_services=["payment-service", "api-gateway", "order-service"],
            severity="P2",
            recommended_action="Increase JVM heap memory limit",
            confidence=0.9,
        ))
        print("Score:", result.observation.grader_score)

Usage (async — for training loops):
    import asyncio
    from client import SREEnvClient
    from models import SREAction

    async def main():
        async with SREEnvClient(base_url="http://localhost:8000") as env:
            result = await env.reset_async(task_id="sre-hard-003")
            result = await env.step_async(SREAction(action_type="query_alerts"))
            result = await env.step_async(SREAction(
                action_type="submit",
                root_cause_service="recommendation-service",
                root_cause_type="configuration_error",
                affected_services=["recommendation-service", "cart-service"],
                severity="P1",
                recommended_action="Rollback feature flag config",
                confidence=0.9,
            ))
            print("Score:", result.observation.grader_score)

    asyncio.run(main())
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SREAction, SREObservation, SREState
except ImportError:
    from models import SREAction, SREObservation, SREState


class SREEnvClient(EnvClient[SREAction, SREObservation, SREState]):
    """Typed WebSocket client for the SRE Incident Investigation environment."""

    def _step_payload(self, action: SREAction) -> Dict:
        payload = {"action_type": action.action_type}
        for field in ["service", "log_level", "time_window_minutes", "log_query",
                      "metric_name", "note", "root_cause_service", "root_cause_type",
                      "affected_services", "severity", "recommended_action", "confidence"]:
            v = getattr(action, field, None)
            if v is not None:
                payload[field] = v
        return payload

    def _parse_result(self, payload: Dict) -> "StepResult[SREObservation]":
        obs_data = payload.get("observation", payload)
        observation = SREObservation(
            action_taken=obs_data.get("action_taken", ""),
            logs=obs_data.get("logs", []),
            metrics=obs_data.get("metrics", []),
            metric_name=obs_data.get("metric_name"),
            alerts=obs_data.get("alerts", []),
            annotation_accepted=obs_data.get("annotation_accepted", False),
            grader_score=obs_data.get("grader_score"),
            grader_breakdown=obs_data.get("grader_breakdown"),
            message=obs_data.get("message", ""),
            queries_remaining=obs_data.get("queries_remaining", 0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> SREState:
        return SREState(
            episode_id=payload.get("episode_id"),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            step_count=payload.get("step_count", 0),
            queries_used=payload.get("queries_used", 0),
            max_queries=payload.get("max_queries", 12),
            annotations=payload.get("annotations", []),
            submitted=payload.get("submitted", False),
            final_score=payload.get("final_score"),
        )
