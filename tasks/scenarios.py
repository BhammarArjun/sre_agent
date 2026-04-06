"""
SRE Incident Task definitions.

Each task defines:
  - Synthetic telemetry (logs, metrics, alerts) seeded for reproducibility
  - A deterministic grader that scores 0.0-1.0
  - Difficulty: easy | medium | hard

Grader criteria and weights:
  root_cause_service  : 0.35
  root_cause_type     : 0.25
  affected_services   : 0.15
  severity            : 0.10
  recommended_action  : 0.15
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0, offset_seconds: int = 0,
        base: Optional[datetime] = None) -> str:
    base = base or datetime(2024, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(minutes=offset_minutes, seconds=offset_seconds)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(ts_offset: int, service: str, level: str, message: str,
         trace_id: Optional[str] = None, ts_seconds: int = 0) -> Dict:
    entry = {
        "timestamp": _ts(ts_offset, ts_seconds),
        "service": service,
        "level": level,
        "message": message,
    }
    if trace_id:
        entry["trace_id"] = trace_id
    return entry


def _metric_series(name: str, values: List[float],
                   start_offset: int = -60, interval_minutes: int = 5) -> List[Dict]:
    return [
        {"timestamp": _ts(start_offset + i * interval_minutes), "value": v}
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    score: float
    breakdown: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {"score": round(self.score, 4), "breakdown": self.breakdown}


def _grade(*, correct_service, correct_type, correct_affected,
           correct_severity, action_keywords,
           submitted_service, submitted_type, submitted_affected,
           submitted_severity, submitted_action) -> GradeResult:

    weights = {
        "root_cause_service": 0.35,
        "root_cause_type":    0.25,
        "affected_services":  0.15,
        "severity":           0.10,
        "recommended_action": 0.15,
    }

    svc_score  = 1.0 if (submitted_service or "").lower() == correct_service.lower() else 0.0
    type_score = 1.0 if (submitted_type or "") == correct_type else 0.0

    if submitted_affected:
        sub = {s.lower() for s in submitted_affected}
        cor = {s.lower() for s in correct_affected}
        if cor:
            p = len(sub & cor) / len(sub) if sub else 0.0
            r = len(sub & cor) / len(cor)
            aff_score = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        else:
            aff_score = 1.0 if not sub else 0.0
    else:
        aff_score = 0.0

    order = ["P1", "P2", "P3", "P4"]
    if submitted_severity == correct_severity:
        sev_score = 1.0
    elif submitted_severity and correct_severity and submitted_severity in order:
        sev_score = 0.5 if abs(order.index(submitted_severity) - order.index(correct_severity)) == 1 else 0.0
    else:
        sev_score = 0.0

    action_text = (submitted_action or "").lower()
    act_score   = sum(1 for kw in action_keywords if kw.lower() in action_text) / len(action_keywords) if action_keywords else 1.0

    scores = dict(root_cause_service=svc_score, root_cause_type=type_score,
                  affected_services=aff_score, severity=sev_score,
                  recommended_action=act_score)
    total = sum(scores[k] * weights[k] for k in weights)
    breakdown = {
        k: {"score": round(scores[k], 4), "weight": weights[k],
            "weighted": round(scores[k]*weights[k], 4)}
        for k in weights
    }
    breakdown["correct_answers"] = dict(
        root_cause_service=correct_service, root_cause_type=correct_type,
        affected_services=correct_affected, severity=correct_severity,
        action_keywords=action_keywords)
    return GradeResult(score=round(total, 4), breakdown=breakdown)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str
    title: str
    description: str
    logs_by_service: Dict[str, List[Dict]] = field(default_factory=dict)
    metrics: Dict[str, List[Dict]]         = field(default_factory=dict)
    alerts: List[Dict]                     = field(default_factory=list)
    _correct_service:  str       = field(default="",   repr=False)
    _correct_type:     str       = field(default="",   repr=False)
    _correct_affected: List[str] = field(default_factory=list, repr=False)
    _correct_severity: str       = field(default="P2", repr=False)
    _action_keywords:  List[str] = field(default_factory=list, repr=False)

    def get_logs(self, service=None, log_level=None,
                 time_window_minutes=30, log_query=None) -> List[Dict]:
        level_order = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        min_idx = level_order.index(log_level) if log_level in level_order else 0
        cutoff  = _ts(-time_window_minutes)
        all_logs: List[Dict] = []
        for svc, entries in self.logs_by_service.items():
            if service and svc.lower() != service.lower():
                continue
            all_logs.extend(entries)
        result = [
            e for e in all_logs
            if e["timestamp"] >= cutoff
            and level_order.index(e["level"]) >= min_idx
            and (not log_query or log_query.lower() in e["message"].lower())
        ]
        return sorted(result, key=lambda e: e["timestamp"])

    def get_metrics(self, metric_name: str) -> List[Dict]:
        return self.metrics.get(metric_name, [])

    def grade(self, submitted_service, submitted_type, submitted_affected,
              submitted_severity, submitted_action) -> GradeResult:
        return _grade(
            correct_service=self._correct_service,
            correct_type=self._correct_type,
            correct_affected=self._correct_affected,
            correct_severity=self._correct_severity,
            action_keywords=self._action_keywords,
            submitted_service=submitted_service,
            submitted_type=submitted_type,
            submitted_affected=submitted_affected,
            submitted_severity=submitted_severity,
            submitted_action=submitted_action,
        )


# ---------------------------------------------------------------------------
# TASK 1 - EASY: payment-service JVM OOM crash loop
# ---------------------------------------------------------------------------

def _build_task_easy() -> Task:
    logs = {
        "api-gateway": [
            _log(-58, "api-gateway", "INFO",  "POST /v1/checkout 200 OK latency=142ms", "t-8801"),
            _log(-55, "api-gateway", "INFO",  "POST /v1/checkout 200 OK latency=138ms", "t-8802"),
            _log(-52, "api-gateway", "INFO",  "POST /v1/checkout 200 OK latency=145ms", "t-8803"),
            _log(-49, "api-gateway", "INFO",  "POST /v1/checkout 200 OK latency=151ms", "t-8804"),
            _log(-46, "api-gateway", "INFO",  "POST /v1/checkout 200 OK latency=149ms", "t-8805"),
            _log(-43, "api-gateway", "WARN",  "POST /v1/checkout 503 upstream=payment-service latency=5012ms", "t-8806"),
            _log(-42, "api-gateway", "WARN",  "POST /v1/checkout 503 upstream=payment-service latency=5011ms", "t-8807"),
            _log(-41, "api-gateway", "ERROR", "POST /v1/checkout 503 upstream=payment-service retries=3/3 exhausted", "t-8808"),
            _log(-40, "api-gateway", "ERROR", "POST /v1/checkout 503 upstream=payment-service retries=3/3 exhausted", "t-8809"),
            _log(-39, "api-gateway", "ERROR", "POST /v1/checkout 503 upstream=payment-service retries=3/3 exhausted", "t-8810"),
            _log(-38, "api-gateway", "WARN",  "Upstream payment-service health probe failed — marking unhealthy"),
            _log(-37, "api-gateway", "ERROR", "POST /v1/checkout 503 upstream=payment-service retries=3/3 exhausted", "t-8811"),
            _log(-36, "api-gateway", "ERROR", "POST /v1/checkout 503 upstream=payment-service retries=3/3 exhausted", "t-8812"),
            _log(-35, "api-gateway", "ERROR", "Circuit breaker OPEN for upstream=payment-service (threshold: 50% errors in 60s)"),
            _log(-34, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN short-circuiting", "t-8813"),
            _log(-32, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN short-circuiting", "t-8814"),
            _log(-30, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN short-circuiting", "t-8815"),
            _log(-28, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN short-circuiting", "t-8816"),
            _log(-25, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN short-circuiting", "t-8817"),
            _log(-22, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN short-circuiting", "t-8818"),
            _log(-18, "api-gateway", "INFO",  "Circuit breaker half-open — probing payment-service"),
            _log(-18, "api-gateway", "ERROR", "Probe failed — circuit breaker returning to OPEN", None, 6),
            _log(-15, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN", "t-8819"),
            _log(-12, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN", "t-8820"),
            _log(-10, "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN", "t-8821"),
            _log(-8,  "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN", "t-8822"),
            _log(-5,  "api-gateway", "ERROR", "POST /v1/checkout 503 circuit_breaker=OPEN", "t-8823"),
        ],
        "payment-service": [
            _log(-60, "payment-service", "INFO",  "PaymentProcessor ready — JVM heap_max=2048MB heap_used=820MB (40%) gc=G1GC"),
            _log(-58, "payment-service", "INFO",  "Processed req_id=PAY-44201 amount=$99.99 card=****4242 status=SUCCESS latency=88ms"),
            _log(-56, "payment-service", "INFO",  "Processed req_id=PAY-44202 amount=$249.00 card=****1337 status=SUCCESS latency=91ms"),
            _log(-54, "payment-service", "INFO",  "Processed req_id=PAY-44203 amount=$19.99 card=****9981 status=SUCCESS latency=85ms"),
            _log(-52, "payment-service", "INFO",  "Processed req_id=PAY-44204 amount=$149.50 card=****5566 status=SUCCESS latency=93ms"),
            _log(-50, "payment-service", "INFO",  "GC stats: minor_gc=12ms major_gc=0ms heap_used=940MB (46%) live_objects=2.1M"),
            _log(-48, "payment-service", "INFO",  "Processed req_id=PAY-44205 amount=$79.99 card=****7723 status=SUCCESS latency=95ms"),
            _log(-46, "payment-service", "WARN",  "JVM heap pressure rising: heap_used=1420MB (69%) — GC overhead increasing"),
            _log(-45, "payment-service", "WARN",  "GC pause: stop-the-world=340ms heap_before=1420MB heap_after=1180MB — not fully reclaimed"),
            _log(-44, "payment-service", "WARN",  "heap_used=1580MB (77%) — possible memory leak in PaymentCache (size=142,000 entries)"),
            _log(-43, "payment-service", "WARN",  "PaymentCache eviction not keeping pace with inserts. Cache size unbounded."),
            _log(-42, "payment-service", "WARN",  "heap_used=1780MB (87%) — GC running continuously, throughput degrading"),
            _log(-41, "payment-service", "WARN",  "GC pause: stop-the-world=1840ms heap_before=1780MB heap_after=1720MB — GC ineffective"),
            _log(-40, "payment-service", "ERROR", "heap_used=1970MB (96%) — CRITICAL: approaching max heap"),
            _log(-40, "payment-service", "ERROR", "java.lang.OutOfMemoryError: Java heap space\n\tat com.payments.cache.PaymentCache.put(PaymentCache.java:218)\n\tat com.payments.processor.PaymentProcessor.process(PaymentProcessor.java:441)", None, 20),
            _log(-40, "payment-service", "FATAL", "Unrecoverable JVM state — OutOfMemoryError in critical thread pool. Shutting down.", None, 30),
            _log(-39, "payment-service", "INFO",  "Pod payment-service-7d9f8b-xk2pq restarted by Kubernetes (reason: OOMKilled) restart_count=1"),
            _log(-39, "payment-service", "INFO",  "PaymentProcessor starting — heap_max=2048MB heap_used=210MB (10%) restart_count=1", None, 15),
            _log(-38, "payment-service", "INFO",  "Processed req_id=PAY-44206 amount=$59.99 status=SUCCESS latency=90ms"),
            _log(-37, "payment-service", "WARN",  "heap_used=1210MB (59%) — heap growing rapidly post-restart"),
            _log(-36, "payment-service", "WARN",  "heap_used=1640MB (80%) — PaymentCache re-warming too aggressively after restart"),
            _log(-35, "payment-service", "ERROR", "heap_used=1950MB (95%) — OOM imminent"),
            _log(-35, "payment-service", "ERROR", "java.lang.OutOfMemoryError: Java heap space\n\tat com.payments.cache.PaymentCache.put(PaymentCache.java:218)", None, 20),
            _log(-35, "payment-service", "FATAL", "Unrecoverable JVM state — shutting down restart_count=2", None, 25),
            _log(-34, "payment-service", "INFO",  "Pod restarted by Kubernetes (OOMKilled) restart_count=2"),
            _log(-34, "payment-service", "INFO",  "PaymentProcessor starting — restart_count=2", None, 15),
            _log(-33, "payment-service", "WARN",  "heap_used=1380MB (67%) — same leak pattern post-restart"),
            _log(-32, "payment-service", "WARN",  "heap_used=1750MB (85%) — GC stop-the-world=2100ms"),
            _log(-31, "payment-service", "ERROR", "java.lang.OutOfMemoryError: Java heap space\n\tat com.payments.cache.PaymentCache.put(PaymentCache.java:218)"),
            _log(-31, "payment-service", "FATAL", "Unrecoverable JVM state — shutting down restart_count=3", None, 10),
            _log(-30, "payment-service", "INFO",  "Pod restarted (OOMKilled) restart_count=3. Kubernetes CrashLoopBackOff active."),
            _log(-29, "payment-service", "WARN",  "CrashLoopBackOff — next restart in 30s (exponential backoff)"),
            _log(-27, "payment-service", "INFO",  "PaymentProcessor starting — restart_count=3 (after backoff)"),
            _log(-26, "payment-service", "WARN",  "heap_used=1550MB (76%) immediately on startup — unbounded cache warming on boot"),
            _log(-25, "payment-service", "ERROR", "java.lang.OutOfMemoryError: Java heap space"),
            _log(-25, "payment-service", "FATAL", "Unrecoverable JVM state — shutting down restart_count=4", None, 30),
            _log(-24, "payment-service", "INFO",  "Pod restarted (OOMKilled) restart_count=4 — Kubernetes backoff now 60s"),
            _log(-20, "payment-service", "WARN",  "Kubernetes will not restart for 60s (exponential backoff, restart_count=4)"),
            _log(-10, "payment-service", "INFO",  "Pod restarting after backoff — restart_count=5"),
            _log(-10, "payment-service", "WARN",  "heap_used=1410MB (69%) — identical leak pattern. PaymentCache has no max size configured.", None, 20),
            _log(-9,  "payment-service", "ERROR", "java.lang.OutOfMemoryError: Java heap space"),
            _log(-9,  "payment-service", "FATAL", "Unrecoverable JVM state — restart_count=5", None, 15),
            _log(-8,  "payment-service", "INFO",  "Pod restarted (OOMKilled) restart_count=5. Kubernetes backoff=120s."),
        ],
        "order-service": [
            _log(-58, "order-service", "INFO",  "Order #ORD-88901 created user_id=u-4421 items=3 total=$99.99 — awaiting payment confirmation"),
            _log(-55, "order-service", "INFO",  "Order #ORD-88901 payment confirmed — status=CONFIRMED fulfillment=queued"),
            _log(-53, "order-service", "INFO",  "Order #ORD-88902 created user_id=u-4422 items=1 total=$249.00 — awaiting payment"),
            _log(-51, "order-service", "INFO",  "Order #ORD-88902 payment confirmed — status=CONFIRMED"),
            _log(-45, "order-service", "INFO",  "Order #ORD-88904 created user_id=u-4424 items=2 total=$149.50 — awaiting payment"),
            _log(-43, "order-service", "WARN",  "Order #ORD-88904 payment callback timeout 5000ms — retrying (1/3)"),
            _log(-43, "order-service", "WARN",  "Order #ORD-88904 payment callback timeout 5000ms — retrying (2/3)", None, 10),
            _log(-43, "order-service", "ERROR", "Order #ORD-88904 payment failed after 3 retries — status=PAYMENT_FAILED", None, 20),
            _log(-42, "order-service", "ERROR", "Order #ORD-88905 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-41, "order-service", "ERROR", "Order #ORD-88906 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-40, "order-service", "ERROR", "Order #ORD-88907 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-39, "order-service", "ERROR", "Order #ORD-88908 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-37, "order-service", "ERROR", "7 orders failed in last 5min due to payment-service unavailability"),
            _log(-35, "order-service", "ERROR", "Order #ORD-88910 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-30, "order-service", "ERROR", "Order #ORD-88911 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-28, "order-service", "WARN",  "Dead-letter queue: 14 failed payment callbacks pending retry. Revenue at risk: $2,841"),
            _log(-25, "order-service", "ERROR", "Order #ORD-88913 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-20, "order-service", "WARN",  "Dead-letter queue: 22 pending payment retries. Estimated revenue at risk: $4,218"),
            _log(-15, "order-service", "ERROR", "Order #ORD-88916 payment-service unavailable — status=PAYMENT_FAILED"),
            _log(-10, "order-service", "WARN",  "Dead-letter queue: 31 pending payment retries. Estimated revenue at risk: $6,441"),
            _log(-5,  "order-service", "WARN",  "Dead-letter queue: 38 pending payment retries. Estimated revenue at risk: $7,892"),
        ],
        "inventory-service": [
            _log(-58, "inventory-service", "INFO",  "Reserved stock: order=#ORD-88901 sku=SKU-4421 qty=1 warehouse=SEA-01"),
            _log(-55, "inventory-service", "INFO",  "Reservation confirmed (payment ok): order=#ORD-88901"),
            _log(-53, "inventory-service", "INFO",  "Reserved stock: order=#ORD-88902 sku=SKU-8821 qty=1 warehouse=SEA-01"),
            _log(-51, "inventory-service", "INFO",  "Reservation confirmed (payment ok): order=#ORD-88902"),
            _log(-43, "inventory-service", "INFO",  "Reserved stock: order=#ORD-88904 sku=SKU-2211 qty=2 warehouse=PDX-01"),
            _log(-43, "inventory-service", "WARN",  "Payment failed for #ORD-88904 — releasing reservation after 5min hold", None, 30),
            _log(-40, "inventory-service", "WARN",  "Multiple reservation releases: orders=[88905,88906,88907] — payment-service down"),
            _log(-35, "inventory-service", "WARN",  "High reservation release rate: 7 in 10min (normal: <1/hr) — payment-service outage"),
            _log(-20, "inventory-service", "WARN",  "22 reservation releases since 13:17 UTC — stock re-pooled but revenue lost"),
            _log(-5,  "inventory-service", "WARN",  "38 reservation releases total — payment outage impacting inventory cycle"),
        ],
        "notification-service": [
            _log(-43, "notification-service", "WARN",  "Order #ORD-88904 confirmation delayed — payment pending"),
            _log(-40, "notification-service", "ERROR", "Failed to send confirmation for #ORD-88904 — payment_status=FAILED"),
            _log(-38, "notification-service", "ERROR", "Failed to send confirmation for #ORD-88905 — payment_status=FAILED"),
            _log(-35, "notification-service", "WARN",  "Email queue: 8 failed-order notifications pending"),
            _log(-20, "notification-service", "WARN",  "Email queue: 21 failed-order notifications. 14 support tickets opened by customers."),
            _log(-10, "notification-service", "WARN",  "Email queue: 32 failed-order notifications. Customer support queue growing."),
        ],
    }

    metrics = {
        "memory_usage": _metric_series("memory_usage",
            [40, 46, 52, 60, 69, 77, 87, 96, 10, 59, 80, 95, 10, 67, 85, 10, 10],
            start_offset=-80, interval_minutes=5),
        "error_rate": _metric_series("error_rate",
            [0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 1.2, 28.0, 52.0, 54.0, 51.0, 53.0, 55.0, 54.0, 52.0, 51.0, 53.0],
            start_offset=-80, interval_minutes=5),
        "request_rate": _metric_series("request_rate",
            [118, 120, 122, 119, 121, 118, 98, 42, 15, 12, 11, 10, 10, 9, 9, 9, 9],
            start_offset=-80, interval_minutes=5),
        "latency_p99": _metric_series("latency_p99",
            [145, 148, 151, 149, 152, 155, 820, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
            start_offset=-80, interval_minutes=5),
        "latency_p50": _metric_series("latency_p50",
            [88, 90, 91, 89, 92, 94, 320, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
            start_offset=-80, interval_minutes=5),
        "cpu_usage": _metric_series("cpu_usage",
            [28, 30, 32, 35, 40, 48, 75, 88, 15, 72, 84, 95, 12, 70, 88, 12, 12],
            start_offset=-80, interval_minutes=5),
        "db_connections": _metric_series("db_connections",
            [42, 43, 44, 43, 44, 45, 42, 38, 35, 36, 35, 34, 35, 34, 34, 33, 33],
            start_offset=-80, interval_minutes=5),
        "cache_hit_rate": _metric_series("cache_hit_rate",
            [91, 91, 91, 90, 90, 90, 89, 88, 88, 88, 87, 87, 87, 86, 86, 86, 85],
            start_offset=-80, interval_minutes=5),
    }

    alerts = [
        {"alert_name": "PodOOMKilled",      "service": "payment-service",
         "severity": "critical", "fired_at": _ts(-40),
         "message": "payment-service OOMKilled 5x in 40 minutes. restart_count=5. CrashLoopBackOff active.",
         "status": "firing"},
        {"alert_name": "HighErrorRate",     "service": "api-gateway",
         "severity": "critical", "fired_at": _ts(-37),
         "message": "api-gateway error rate 52% (threshold 5%). Upstream: payment-service.",
         "status": "firing"},
        {"alert_name": "CircuitBreakerOpen","service": "api-gateway",
         "severity": "warning",  "fired_at": _ts(-35),
         "message": "Circuit breaker OPEN for payment-service. Checkout short-circuiting.",
         "status": "firing"},
        {"alert_name": "RevenueImpact",     "service": "order-service",
         "severity": "critical", "fired_at": _ts(-20),
         "message": "38 orders stuck in PAYMENT_FAILED. Estimated revenue at risk: $7,892.",
         "status": "firing"},
    ]

    return Task(
        task_id="sre-easy-001",
        difficulty="easy",
        title="Checkout Failures — Payment Service OOMKilled (CrashLoopBackOff)",
        description=(
            "INCIDENT ALERT — P2 — 13:20 UTC\n\n"
            "Customer-facing checkout failing. Error rate on POST /v1/checkout\n"
            "spiked to ~52%. Customers cannot complete purchases.\n"
            "Finance flagging revenue impact. On-call SRE paged.\n\n"
            "System topology:\n"
            "  api-gateway → payment-service  (card processing)\n"
            "  api-gateway → order-service → payment-service\n"
            "  order-service → inventory-service\n"
            "  order-service → notification-service\n\n"
            "Query logs, metrics, and alerts to identify root cause.\n"
            "Then submit your structured incident report.\n\n"
            "Available services: api-gateway, payment-service, order-service,\n"
            "  inventory-service, notification-service\n"
            "Available metrics: error_rate, latency_p99, latency_p50, cpu_usage,\n"
            "  memory_usage, db_connections, request_rate, cache_hit_rate"
        ),
        logs_by_service=logs,
        metrics=metrics,
        alerts=alerts,
        _correct_service="payment-service",
        _correct_type="resource_exhaustion",
        _correct_affected=["payment-service", "api-gateway", "order-service"],
        _correct_severity="P2",
        _action_keywords=["memory", "heap", "limit", "jvm"],
    )


# ---------------------------------------------------------------------------
# TASK 2 - MEDIUM: analytics exhausts shared DB connection pool
# ---------------------------------------------------------------------------

def _build_task_medium() -> Task:
    logs = {
        "api-gateway": [
            _log(-75, "api-gateway", "INFO",  "All upstreams healthy. Serving 840 req/min. p99=198ms"),
            _log(-60, "api-gateway", "INFO",  "GET /api/v2/orders 200 OK latency=198ms"),
            _log(-55, "api-gateway", "INFO",  "GET /api/v2/orders 200 OK latency=201ms"),
            _log(-50, "api-gateway", "INFO",  "GET /api/v2/orders 200 OK latency=195ms"),
            _log(-48, "api-gateway", "WARN",  "GET /api/v2/orders 200 OK latency=890ms — upstream degrading"),
            _log(-45, "api-gateway", "WARN",  "GET /api/v2/orders 200 OK latency=1820ms — upstream slow"),
            _log(-43, "api-gateway", "WARN",  "GET /api/v2/orders 200 OK latency=4200ms — near timeout"),
            _log(-42, "api-gateway", "ERROR", "GET /api/v2/orders 504 Gateway Timeout upstream=order-service timeout=5000ms"),
            _log(-41, "api-gateway", "ERROR", "GET /api/v2/orders 504 Gateway Timeout upstream=order-service"),
            _log(-40, "api-gateway", "ERROR", "GET /api/v2/orders 504 Gateway Timeout upstream=order-service (3 consecutive)"),
            _log(-39, "api-gateway", "WARN",  "Retrying order-service (attempt 1/3) — upstream not responding"),
            _log(-38, "api-gateway", "ERROR", "GET /api/v2/orders 504 — retry 1/3 also timed out"),
            _log(-37, "api-gateway", "ERROR", "GET /api/v2/orders 503 order-service marked unhealthy by load balancer"),
            _log(-36, "api-gateway", "ERROR", "Circuit breaker OPENING for order-service error_rate=61% threshold=50%"),
            _log(-35, "api-gateway", "ERROR", "GET /api/v2/orders 503 circuit_breaker=OPEN upstream=order-service"),
            _log(-30, "api-gateway", "ERROR", "GET /api/v2/orders 503 circuit_breaker=OPEN upstream=order-service"),
            _log(-25, "api-gateway", "ERROR", "GET /api/v2/orders 503 circuit_breaker=OPEN upstream=order-service"),
            _log(-20, "api-gateway", "ERROR", "GET /api/v2/orders 503 circuit_breaker=OPEN upstream=order-service"),
            _log(-18, "api-gateway", "INFO",  "Circuit breaker half-open — probing order-service"),
            _log(-18, "api-gateway", "ERROR", "Probe failed — circuit returning to OPEN (order-service still unhealthy)", None, 5),
            _log(-15, "api-gateway", "ERROR", "GET /api/v2/orders 503 circuit_breaker=OPEN"),
            _log(-10, "api-gateway", "INFO",  "DB connection alert resolved — analytics-service released connections"),
            _log(-8,  "api-gateway", "INFO",  "Circuit breaker half-open — reprobing order-service"),
            _log(-7,  "api-gateway", "INFO",  "order-service responding — circuit breaker CLOSING"),
            _log(-5,  "api-gateway", "INFO",  "GET /api/v2/orders 200 OK latency=205ms — service recovered"),
        ],
        "order-service": [
            _log(-75, "order-service", "INFO",  "Healthy. db_pool=8/50 threads=24/200"),
            _log(-70, "order-service", "INFO",  "Order #ORD-77001 fetched user_id=u-8812 items=4 db_query=12ms"),
            _log(-65, "order-service", "INFO",  "Order #ORD-77002 created user_id=u-8813 total=$189.00"),
            _log(-60, "order-service", "INFO",  "db_pool=12/50 threads=28/200 — normal"),
            _log(-55, "order-service", "INFO",  "Order #ORD-77003 fetched db_query=14ms"),
            _log(-52, "order-service", "WARN",  "db_pool=28/50 — filling up (normal threshold <20)"),
            _log(-50, "order-service", "WARN",  "db_pool=38/50 — elevated. DB appears saturated."),
            _log(-48, "order-service", "WARN",  "db_pool=44/50 — connections not releasing."),
            _log(-46, "order-service", "ERROR", "db_pool=50/50 — POOL EXHAUSTED. Requests queuing."),
            _log(-45, "order-service", "ERROR", "DB connection acquisition timeout 5000ms — request_id=r-9901 order=#ORD-77004"),
            _log(-44, "order-service", "ERROR", "DB connection acquisition timeout 5000ms — request_id=r-9902 order=#ORD-77005"),
            _log(-43, "order-service", "ERROR", "DB connection acquisition timeout 5000ms — request_id=r-9903 order=#ORD-77006"),
            _log(-42, "order-service", "ERROR", "DB connection acquisition timeout 5000ms — request_id=r-9904 order=#ORD-77007"),
            _log(-41, "order-service", "ERROR", "Failed to process #ORD-77005: HikariPool.getConnection — pool exhausted"),
            _log(-40, "order-service", "ERROR", "Failed to process #ORD-77006: HikariPool.getConnection — pool exhausted"),
            _log(-39, "order-service", "ERROR", "Failed to process #ORD-77007: HikariPool.getConnection — pool exhausted"),
            _log(-38, "order-service", "ERROR", "5 requests failed in 60s — DB connection exhaustion"),
            _log(-37, "order-service", "ERROR", "Health check /health cannot reach postgres — returning 503"),
            _log(-36, "order-service", "FATAL", "Kubernetes liveness probe failed 3x consecutively — pod marked unhealthy"),
            _log(-35, "order-service", "INFO",  "Pod order-service-6c8d9b-mn4rs restarting (liveness probe failure)"),
            _log(-34, "order-service", "INFO",  "Pod starting up — attempting DB connection"),
            _log(-34, "order-service", "ERROR", "DB connection pool exhausted IMMEDIATELY on startup — postgres at limit", None, 10),
            _log(-33, "order-service", "FATAL", "Cannot initialise — DB unavailable. Pod crashing.", None, 20),
            _log(-32, "order-service", "INFO",  "Pod restarting (CrashLoopBackOff)"),
            _log(-30, "order-service", "INFO",  "Pod starting (backoff=30s)"),
            _log(-30, "order-service", "ERROR", "DB connection pool exhausted on startup — postgres still at limit", None, 5),
            _log(-25, "order-service", "ERROR", "Still cannot acquire DB connections — postgres connection limit"),
            _log(-20, "order-service", "WARN",  "Kubernetes CrashLoopBackOff — pod will retry in 60s"),
            _log(-10, "order-service", "INFO",  "Postgres connections freed — pool recovering db_pool=5/50"),
            _log(-9,  "order-service", "INFO",  "DB connection acquired successfully — service starting normally"),
            _log(-8,  "order-service", "INFO",  "Healthy — db_pool=8/50 all endpoints responding"),
        ],
        "analytics-service": [
            _log(-80, "analytics-service", "INFO",  "Nightly revenue report job scheduled — cron='0 13 * * *' job_id=report-2024-06-15"),
            _log(-78, "analytics-service", "INFO",  "Connecting to postgres for report job. Acquiring connection pool."),
            _log(-78, "analytics-service", "INFO",  "Running SQL: SELECT o.*, oi.*, p.*, u.* FROM orders o JOIN order_items oi ON o.id=oi.order_id JOIN products p ON oi.product_id=p.id JOIN users u ON o.user_id=u.id WHERE o.created_at > NOW() - INTERVAL '90 days'  -- WARNING: no LIMIT clause"),
            _log(-75, "analytics-service", "INFO",  "Query running — estimated 4.2M rows. elapsed=2m rows_fetched=180,000"),
            _log(-72, "analytics-service", "WARN",  "Query running — elapsed=6m rows_fetched=820,000. Spawning parallel workers."),
            _log(-70, "analytics-service", "INFO",  "Parallel workers spawned: 8 workers x 4 conns = 32 DB connections held"),
            _log(-67, "analytics-service", "WARN",  "Query running — elapsed=11m rows_fetched=1,840,000. 38 DB connections held."),
            _log(-65, "analytics-service", "WARN",  "Aggregation phase: sorting 1.84M rows — spilling to disk (work_mem=64MB insufficient)"),
            _log(-62, "analytics-service", "WARN",  "Query running — elapsed=16m rows_fetched=2,610,000. 44 DB connections held."),
            _log(-60, "analytics-service", "WARN",  "Query running — elapsed=18m rows_fetched=3,100,000. 48 DB connections held."),
            _log(-58, "analytics-service", "WARN",  "Query running — elapsed=20m rows_fetched=3,580,000. 50 DB connections held. Pool near limit."),
            _log(-55, "analytics-service", "WARN",  "Postgres max_connections approaching — other services may be starved"),
            _log(-52, "analytics-service", "WARN",  "Query running — elapsed=26m rows_fetched=4,020,000. Still holding 50 connections."),
            _log(-48, "analytics-service", "WARN",  "Query running — elapsed=30m rows_fetched=4,200,000 (all rows fetched). Aggregation in progress."),
            _log(-45, "analytics-service", "WARN",  "Aggregation running — computing GROUP BY across 4.2M rows. 50 connections still locked."),
            _log(-40, "analytics-service", "WARN",  "Aggregation running — elapsed=38m. All 50 DB connections locked by this job."),
            _log(-35, "analytics-service", "WARN",  "Query running — elapsed=43m. Downstream services reporting connection errors."),
            _log(-30, "analytics-service", "WARN",  "Query running — elapsed=48m. order-service CrashLoopBackOff."),
            _log(-25, "analytics-service", "WARN",  "Query running — elapsed=53m. api-gateway circuit breaker open."),
            _log(-12, "analytics-service", "INFO",  "Query complete — elapsed=66m rows=4,218,441 report_rows=28,441. Releasing connections."),
            _log(-11, "analytics-service", "INFO",  "All 50 DB connections released back to pool"),
            _log(-10, "analytics-service", "INFO",  "Report written to s3://analytics-reports/revenue-2024-06-15.csv size=142MB"),
            _log(-9,  "analytics-service", "INFO",  "Job report-2024-06-15 COMPLETE. Duration: 69 minutes."),
        ],
        "postgres": [
            _log(-80, "postgres", "INFO",  "Active connections: 14/100 — healthy"),
            _log(-78, "postgres", "INFO",  "New connections from analytics-service. Active: 15/100"),
            _log(-75, "postgres", "INFO",  "Active connections: 24/100 — analytics job ramping"),
            _log(-70, "postgres", "INFO",  "Active connections: 46/100 — analytics parallel workers"),
            _log(-67, "postgres", "WARN",  "Active connections: 68/100 — approaching advisory threshold"),
            _log(-65, "postgres", "WARN",  "Active connections: 78/100 — high utilisation"),
            _log(-62, "postgres", "WARN",  "Active connections: 88/100 — WARNING: saturation risk"),
            _log(-60, "postgres", "WARN",  "Active connections: 96/100 — CRITICAL: 4 superuser slots remaining"),
            _log(-58, "postgres", "ERROR", "FATAL: remaining connection slots reserved for non-replication superuser connections"),
            _log(-58, "postgres", "ERROR", "Connection rejected for order-service — max_connections=100 reached", None, 5),
            _log(-57, "postgres", "ERROR", "Connection rejected for order-service — max_connections=100 reached"),
            _log(-56, "postgres", "ERROR", "Connection rejected for order-service — max_connections=100 reached"),
            _log(-50, "postgres", "ERROR", "Multiple rejections/sec — all application slots occupied by analytics job"),
            _log(-45, "postgres", "ERROR", "Lock wait timeout on table=orders: analytics holding table-level read lock"),
            _log(-40, "postgres", "ERROR", "Connection rejected for order-service (startup probe) — still saturated"),
            _log(-35, "postgres", "WARN",  "Active connections: 98/100 — analytics: 94 conns, order-service: 4 conns"),
            _log(-12, "postgres", "INFO",  "Active connections: 98 → 14 — analytics job released 84 connections"),
            _log(-11, "postgres", "INFO",  "Active connections: 14/100 — healthy"),
            _log(-10, "postgres", "INFO",  "order-service reconnecting — all requests accepted"),
        ],
        "notification-service": [
            _log(-45, "notification-service", "WARN",  "Order status fetch failed — order-service 504. Delaying shipment alerts."),
            _log(-40, "notification-service", "ERROR", "Cannot fetch order details: GET /api/v2/orders/77004 → 503"),
            _log(-35, "notification-service", "ERROR", "Cannot fetch order details: GET /api/v2/orders/77005 → 503"),
            _log(-30, "notification-service", "WARN",  "Notification queue: 340 pending shipment alerts"),
            _log(-20, "notification-service", "WARN",  "Notification queue: 820 pending. Customers waiting for confirmations."),
            _log(-10, "notification-service", "INFO",  "order-service recovering — draining queue (820 pending)"),
            _log(-5,  "notification-service", "INFO",  "Queue drain: 820 → 610 sent"),
        ],
    }

    metrics = {
        "db_connections": _metric_series("db_connections",
            [14, 18, 24, 38, 52, 68, 82, 96, 98, 98, 98, 98, 98, 98, 98, 14, 14],
            start_offset=-80, interval_minutes=5),
        "latency_p99": _metric_series("latency_p99",
            [198, 205, 210, 420, 1200, 4800, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 4900, 4800, 210, 205],
            start_offset=-80, interval_minutes=5),
        "error_rate": _metric_series("error_rate",
            [0.1, 0.2, 0.3, 0.5, 2.1, 12.0, 42.0, 55.0, 58.0, 56.0, 54.0, 55.0, 57.0, 55.0, 54.0, 0.8, 0.2],
            start_offset=-80, interval_minutes=5),
        "request_rate": _metric_series("request_rate",
            [840, 842, 838, 830, 812, 780, 640, 480, 410, 390, 380, 372, 368, 362, 355, 820, 839],
            start_offset=-80, interval_minutes=5),
        "cpu_usage": _metric_series("cpu_usage",
            [24, 26, 28, 35, 42, 55, 68, 78, 82, 84, 85, 84, 83, 82, 80, 28, 25],
            start_offset=-80, interval_minutes=5),
        "memory_usage": _metric_series("memory_usage",
            [54, 55, 56, 58, 62, 68, 72, 74, 75, 75, 75, 74, 74, 73, 72, 57, 55],
            start_offset=-80, interval_minutes=5),
        "cache_hit_rate": _metric_series("cache_hit_rate",
            [89, 89, 88, 84, 74, 60, 48, 40, 36, 34, 33, 32, 32, 31, 31, 86, 88],
            start_offset=-80, interval_minutes=5),
        "latency_p50": _metric_series("latency_p50",
            [92, 95, 98, 210, 840, 3600, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 4900, 4800, 98, 94],
            start_offset=-80, interval_minutes=5),
    }

    alerts = [
        {"alert_name": "DBConnectionsWarning", "service": "postgres",
         "severity": "warning",  "fired_at": _ts(-62),
         "message": "PostgreSQL connections 88/100 (88%). Threshold: 80%.",
         "status": "resolved"},
        {"alert_name": "DBConnectionsCritical","service": "postgres",
         "severity": "critical", "fired_at": _ts(-58),
         "message": "PostgreSQL at max_connections (98/100). New connections rejected.",
         "status": "resolved"},
        {"alert_name": "HighLatency",          "service": "api-gateway",
         "severity": "warning",  "fired_at": _ts(-45),
         "message": "p99 latency on /api/v2/orders exceeded 1s (current: 4.8s).",
         "status": "resolved"},
        {"alert_name": "ServiceUnhealthy",     "service": "order-service",
         "severity": "critical", "fired_at": _ts(-36),
         "message": "order-service liveness probe failing — CrashLoopBackOff.",
         "status": "resolved"},
        {"alert_name": "HighErrorRate",        "service": "api-gateway",
         "severity": "critical", "fired_at": _ts(-35),
         "message": "api-gateway error rate 55% on /api/v2/orders. Circuit breaker OPEN.",
         "status": "resolved"},
        {"alert_name": "LongRunningQuery",     "service": "analytics-service",
         "severity": "warning",  "fired_at": _ts(-65),
         "message": "analytics-service SQL query running >10 minutes. 38 DB connections held.",
         "status": "resolved"},
    ]

    return Task(
        task_id="sre-medium-002",
        difficulty="medium",
        title="Order Service Outage — DB Connection Pool Exhausted by Analytics Job",
        description=(
            "INCIDENT ALERT — P2 — 13:25 UTC\n\n"
            "Order lookups and checkout degraded. api-gateway reporting ~55%\n"
            "error rate on /api/v2/orders. order-service in CrashLoopBackOff.\n"
            "Multiple services showing symptoms. Outage now resolving, but\n"
            "you need to identify the ROOT CAUSE — not just the loudest victim.\n\n"
            "System topology:\n"
            "  api-gateway → order-service → postgres (shared pool, 100 max)\n"
            "  analytics-service → postgres (same shared pool)\n"
            "  order-service → notification-service (async alerts)\n\n"
            "Several services look broken. Find which one CAUSED the incident.\n"
            "Hint: the guilty service may look healthy in some metrics.\n\n"
            "Available services: api-gateway, order-service, analytics-service,\n"
            "  notification-service, postgres\n"
            "Available metrics: error_rate, latency_p99, latency_p50, cpu_usage,\n"
            "  memory_usage, db_connections, request_rate, cache_hit_rate"
        ),
        logs_by_service=logs,
        metrics=metrics,
        alerts=alerts,
        _correct_service="analytics-service",
        _correct_type="resource_exhaustion",
        _correct_affected=["analytics-service", "order-service", "api-gateway", "notification-service", "postgres"],
        _correct_severity="P2",
        _action_keywords=["connection", "pool", "query", "limit", "analytics"],
    )


# ---------------------------------------------------------------------------
# TASK 3 - HARD: silent revenue loss from bad feature flag
# ---------------------------------------------------------------------------

def _build_task_hard() -> Task:
    logs = {
        "config-service": [
            _log(-130, "config-service", "INFO",  "Config push initiated — job_id=CI-4821 commit=a3f92c1 author=deploy-bot branch=main"),
            _log(-130, "config-service", "INFO",  "Validating config schema for recommendation-service...", None, 5),
            _log(-130, "config-service", "INFO",  "Schema validation PASSED (no breaking change detected by linter)", None, 10),
            _log(-129, "config-service", "INFO",  "Pushing to recommendation-service: feature_flags.use_v2_product_ids=true (was: false)"),
            _log(-129, "config-service", "INFO",  "recommendation-service ACK in 240ms — config applied", None, 5),
            _log(-129, "config-service", "INFO",  "Config push complete — job_id=CI-4821 services_updated=1 duration=14s", None, 10),
            _log(-60,  "config-service", "INFO",  "Routine config sync — no changes. All services up-to-date."),
            _log(-30,  "config-service", "INFO",  "Routine config sync — no changes. All services up-to-date."),
        ],
        "recommendation-service": [
            _log(-135, "recommendation-service", "INFO",  "Serving recommendations — model=collab-filter-v3 avg_latency=28ms rps=210"),
            _log(-130, "recommendation-service", "INFO",  "Config update received: feature_flags.use_v2_product_ids=true (was: false)"),
            _log(-130, "recommendation-service", "INFO",  "Switching product ID schema: integer IDs → 'PRD-' prefixed string IDs", None, 5),
            _log(-130, "recommendation-service", "INFO",  "ID format change active: 4421 → 'PRD-4421', 8821 → 'PRD-8821', etc.", None, 8),
            _log(-129, "recommendation-service", "INFO",  "Serving recs — req_id=rec-10021 user=u-5512 products=['PRD-4421','PRD-8821','PRD-2291']"),
            _log(-128, "recommendation-service", "INFO",  "Serving recs — req_id=rec-10022 user=u-5513 products=['PRD-9912','PRD-3314']"),
            _log(-125, "recommendation-service", "INFO",  "Serving recs — req_id=rec-10023 user=u-5514 products=['PRD-4421','PRD-7712']"),
            _log(-120, "recommendation-service", "INFO",  "rps=212 avg_latency=29ms — all metrics nominal"),
            _log(-100, "recommendation-service", "INFO",  "rps=215 avg_latency=28ms — all metrics nominal"),
            _log(-80,  "recommendation-service", "INFO",  "rps=209 avg_latency=30ms — all metrics nominal"),
            _log(-60,  "recommendation-service", "INFO",  "rps=211 avg_latency=29ms — all metrics nominal"),
            _log(-30,  "recommendation-service", "INFO",  "rps=208 avg_latency=28ms — all metrics nominal"),
        ],
        "cart-service": [
            _log(-135, "cart-service", "INFO",  "Cart updated user=u-5510 items=[4421,8821] total=$124.98 catalog=HIT"),
            _log(-134, "cart-service", "INFO",  "Cart updated user=u-5511 items=[9912,3314] total=$67.50 catalog=HIT"),
            _log(-130, "cart-service", "INFO",  "Cart updated user=u-5512 recommended=['PRD-4421','PRD-8821'] — resolving prices"),
            _log(-130, "cart-service", "WARN",  "Product lookup FAILED: id='PRD-4421' not found in catalog (expected integer, got string)", None, 5),
            _log(-130, "cart-service", "WARN",  "Product lookup FAILED: id='PRD-8821' not found in catalog (expected integer, got string)", None, 6),
            _log(-130, "cart-service", "WARN",  "Fallback price $0.00 applied for unresolved product 'PRD-4421' (catalog miss)", None, 7),
            _log(-130, "cart-service", "WARN",  "Fallback price $0.00 applied for unresolved product 'PRD-8821' (catalog miss)", None, 8),
            _log(-130, "cart-service", "INFO",  "Cart saved user=u-5512 recommended=['PRD-4421','PRD-8821'] total=$0.00 (2 items priced $0)", None, 9),
            _log(-129, "cart-service", "WARN",  "Product lookup FAILED: id='PRD-9912' not found in catalog"),
            _log(-129, "cart-service", "WARN",  "Product lookup FAILED: id='PRD-3314' not found in catalog"),
            _log(-129, "cart-service", "INFO",  "Cart saved user=u-5513 recommended=['PRD-9912','PRD-3314'] total=$0.00"),
            _log(-128, "cart-service", "WARN",  "Cache miss rate rising — 'PRD-' prefixed IDs not matching any catalog entries"),
            _log(-125, "cart-service", "WARN",  "Product ID format mismatch count: 18 in last 5min"),
            _log(-120, "cart-service", "WARN",  "Cart total anomaly: 12 carts with $0.00 line items in last 10min"),
            _log(-115, "cart-service", "WARN",  "Product ID format mismatch: 67 in last 15min. All share 'PRD-' prefix."),
            _log(-110, "cart-service", "WARN",  "Avg cart value dropping: was $94.20, now $61.40 (-35%). Recommended items pricing at $0."),
            _log(-105, "cart-service", "WARN",  "Product ID format mismatch: 142 in last 25min"),
            _log(-100, "cart-service", "WARN",  "Avg cart value: $42.10 (-55% from baseline). Recommended items = 100% zero-priced."),
            _log(-90,  "cart-service", "WARN",  "Mismatch count: 298 in last 40min. All 'PRD-' prefixed. Catalog uses integer IDs only."),
            _log(-80,  "cart-service", "WARN",  "Cache hit rate 58% (was 91%). Cache misses caused by unresolvable 'PRD-' IDs."),
            _log(-70,  "cart-service", "WARN",  "Avg cart value: $28.40 (-70% from $94.20 baseline)"),
            _log(-60,  "cart-service", "ERROR", "602 carts in last 70min contain at least one $0.00 item from unresolved PRD- IDs"),
            _log(-45,  "cart-service", "ERROR", "Avg cart value: $18.90. 71% of recommended items priced at $0.00. Revenue bleeding."),
            _log(-30,  "cart-service", "ERROR", "Mismatch count: 1,841 total since 11:50 UTC. All 'PRD-' prefix. Pattern is consistent."),
            _log(-15,  "cart-service", "ERROR", "Revenue impact: 1,288 orders with incorrect totals. Estimated loss: $48,200."),
            _log(-5,   "cart-service", "ERROR", "CRITICAL: avg cart value $11.20 (-88%). Silent revenue loss ongoing for 125 minutes."),
        ],
        "product-catalog": [
            _log(-130, "product-catalog", "INFO",  "Catalog healthy — 48,200 products indexed. All integer IDs. Cache warm."),
            _log(-130, "product-catalog", "WARN",  "Product lookup miss: id='PRD-4421' — format unrecognised (not in catalog)", None, 5),
            _log(-130, "product-catalog", "WARN",  "Product lookup miss: id='PRD-8821' — format unrecognised", None, 6),
            _log(-128, "product-catalog", "WARN",  "18 lookups with unrecognised format in last 2min"),
            _log(-125, "product-catalog", "WARN",  "67 failed lookups — all 'PRD-' prefixed. Catalog only indexes integers."),
            _log(-120, "product-catalog", "WARN",  "Cache miss rate rising: 142 failed lookups. Returning null for unresolved IDs."),
            _log(-110, "product-catalog", "WARN",  "298 failed lookups (PRD- IDs). Returning $0.00 fallback to callers."),
            _log(-100, "product-catalog", "WARN",  "524 failed lookups in 30min. ID schema mismatch — expected int, got string."),
            _log(-80,  "product-catalog", "WARN",  "912 failed lookups total since 11:50 UTC."),
            _log(-60,  "product-catalog", "WARN",  "1,441 failed lookups. Impacting ~71% of recommended-item price resolutions."),
            _log(-30,  "product-catalog", "WARN",  "1,841 failed lookups. recommendation-service passing 'PRD-' IDs; catalog indexes integers only."),
            _log(-5,   "product-catalog", "WARN",  "2,291 failed lookups total since feature flag enabled at 11:50 UTC."),
        ],
        "payment-service": [
            _log(-135, "payment-service", "INFO",  "Processed order=#ORD-99001 amount=$124.98 card=****4242 status=SUCCESS"),
            _log(-134, "payment-service", "INFO",  "Processed order=#ORD-99002 amount=$67.50 card=****8821 status=SUCCESS"),
            _log(-130, "payment-service", "INFO",  "Processed order=#ORD-99003 amount=$0.00 card=****5512 status=SUCCESS — charged $0.00"),
            _log(-129, "payment-service", "INFO",  "Processed order=#ORD-99004 amount=$0.00 card=****5513 status=SUCCESS — charged $0.00"),
            _log(-128, "payment-service", "INFO",  "Processed order=#ORD-99005 amount=$14.99 card=****5514 status=SUCCESS (1 item resolved, 2 at $0)"),
            _log(-120, "payment-service", "WARN",  "Unusual: 8 orders with total < $5.00 in last 10min (baseline: ~0/hr)"),
            _log(-110, "payment-service", "WARN",  "Unusual: 28 orders with total < $5.00 in last 20min"),
            _log(-100, "payment-service", "WARN",  "Unusual: 72 orders with total < $5.00 in last 30min. Revenue metric anomaly."),
            _log(-90,  "payment-service", "WARN",  "Revenue anomaly: $0.00 orders now 18% of volume (was <0.1%)"),
            _log(-80,  "payment-service", "WARN",  "Revenue anomaly: $0.00 orders 31% of volume. Avg order value $28.40 (was $94.20)."),
            _log(-60,  "payment-service", "ERROR", "Revenue anomaly: $0.00 orders 44% of volume. Triggering finance alert."),
            _log(-45,  "payment-service", "ERROR", "Finance alert: hourly revenue $4,210 vs forecast $18,800 (-78%). Escalating to P1."),
            _log(-30,  "payment-service", "ERROR", "1,288 orders processed at incorrect (undercharged) amounts since 11:50 UTC."),
            _log(-15,  "payment-service", "ERROR", "Cumulative revenue loss ~$48,200 in 115 minutes. P1 criteria met."),
        ],
        "api-gateway": [
            _log(-135, "api-gateway", "INFO",  "All services healthy. p99=204ms error_rate=0.2%"),
            _log(-120, "api-gateway", "INFO",  "All services healthy. p99=206ms error_rate=0.2%"),
            _log(-90,  "api-gateway", "INFO",  "All services healthy. p99=208ms error_rate=0.3%"),
            _log(-60,  "api-gateway", "INFO",  "All services healthy. p99=210ms error_rate=0.3%"),
            _log(-30,  "api-gateway", "INFO",  "All services healthy. p99=212ms error_rate=0.4%"),
            _log(-5,   "api-gateway", "INFO",  "All services healthy. p99=214ms error_rate=0.4% — no 5xx errors"),
        ],
    }

    metrics = {
        "error_rate": _metric_series("error_rate",
            [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5],
            start_offset=-130, interval_minutes=10),
        "latency_p99": _metric_series("latency_p99",
            [201, 203, 204, 205, 206, 207, 208, 208, 209, 210, 211, 212, 213, 214],
            start_offset=-130, interval_minutes=10),
        "cache_hit_rate": _metric_series("cache_hit_rate",
            [91, 91, 90, 88, 84, 78, 70, 62, 55, 48, 42, 38, 36, 35],
            start_offset=-130, interval_minutes=10),
        "request_rate": _metric_series("request_rate",
            [838, 842, 840, 844, 841, 838, 836, 834, 832, 830, 829, 828, 826, 824],
            start_offset=-130, interval_minutes=10),
        "cpu_usage": _metric_series("cpu_usage",
            [27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33],
            start_offset=-130, interval_minutes=10),
        "memory_usage": _metric_series("memory_usage",
            [53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60],
            start_offset=-130, interval_minutes=10),
        "db_connections": _metric_series("db_connections",
            [38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45],
            start_offset=-130, interval_minutes=10),
        "latency_p50": _metric_series("latency_p50",
            [86, 87, 88, 88, 89, 90, 90, 91, 91, 92, 93, 93, 94, 94],
            start_offset=-130, interval_minutes=10),
    }

    alerts = [
        {"alert_name": "CacheHitRateDegraded", "service": "cart-service",
         "severity": "warning",  "fired_at": _ts(-90),
         "message": "cart-service cache hit rate fell from 91% to 55% over 40min. Unusual miss pattern on 'PRD-' keys.",
         "status": "firing"},
        {"alert_name": "RevenueAnomaly",        "service": "payment-service",
         "severity": "critical", "fired_at": _ts(-60),
         "message": "Hourly revenue $4,210 vs forecast $18,800 (-78%). High volume of $0.00 orders.",
         "status": "firing"},
        {"alert_name": "FinanceEscalation",     "service": "payment-service",
         "severity": "critical", "fired_at": _ts(-45),
         "message": "Finance escalation: cumulative revenue loss ~$48,200. P1 criteria met. CEO notified.",
         "status": "firing"},
    ]

    return Task(
        task_id="sre-hard-003",
        difficulty="hard",
        title="Silent Revenue Loss — Feature Flag Breaks Product ID Schema",
        description=(
            "INCIDENT ALERT — P1 — Business Impact — 13:57 UTC\n\n"
            "Finance team escalation: revenue past 2 hours is ~78% below forecast.\n"
            "No services are down. No 5xx errors. No SLA alerts breached.\n"
            "Checkout appears to work — orders are completing. But average order\n"
            "value collapsed from $94 to $11. Customers are being undercharged.\n\n"
            "This is a silent, data-corruption class incident.\n"
            "A config change may have been deployed recently — investigate everything.\n\n"
            "System topology:\n"
            "  config-service → recommendation-service (pushes feature flags)\n"
            "  recommendation-service → cart-service (adds recommended items)\n"
            "  cart-service → product-catalog (resolves prices by product ID)\n"
            "  cart-service → payment-service (submits cart total for charge)\n"
            "  api-gateway → all services\n\n"
            "Find what changed, which service is the root cause, and what must\n"
            "be done immediately to stop the ongoing revenue loss.\n\n"
            "Available services: api-gateway, recommendation-service, cart-service,\n"
            "  product-catalog, payment-service, config-service\n"
            "Available metrics: error_rate, latency_p99, latency_p50, cpu_usage,\n"
            "  memory_usage, db_connections, request_rate, cache_hit_rate"
        ),
        logs_by_service=logs,
        metrics=metrics,
        alerts=alerts,
        _correct_service="recommendation-service",
        _correct_type="configuration_error",
        _correct_affected=["recommendation-service", "cart-service", "product-catalog", "payment-service"],
        _correct_severity="P1",
        _action_keywords=["feature flag", "config", "product", "rollback", "revert"],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {
    "sre-easy-001":   _build_task_easy(),
    "sre-medium-002": _build_task_medium(),
    "sre-hard-003":   _build_task_hard(),
}

TASK_IDS_BY_DIFFICULTY = {
    "easy":   ["sre-easy-001"],
    "medium": ["sre-medium-002"],
    "hard":   ["sre-hard-003"],
}

ALL_TASK_IDS = list(TASKS.keys())
