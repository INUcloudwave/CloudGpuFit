"""Instance recommendation: workload SLO -> one single-GPU instance.

Turns per-instance batch curves (predicted latency/throughput at several batch
sizes) plus the user's SLO and budget into a recommendation. Pure functions,
no I/O: main.py calls `recommend()` whenever a job has both a result and an
SLO, so changing the SLO never re-runs prediction.

Decisions (throughput part; evidence in the `script` repo, task 17/18 and
HANDOFF §3.7-§3.9):

  * ONE instance (2026-11-23). The workload is deployed on a single instance;
    the recommendation is which instance type. No multi-unit configurations
    and no mixing of types.
  * Throughput = batch / latency (measured: sustained / single-request
    throughput 0.999-1.006 on all six GPUs).
  * Latency T = the predicted P90 of ONE batch run (batch processing time).
    Only processing time is considered: time spent waiting in a queue or for
    a batch to fill is not modelled (stated as a limitation). The latency
    definition is the latency part's (predictor) and is kept as is.
  * A batch is usable when its predicted P90 <= the latency SLO and it is not
    larger than the batch the user entered (VRAM was checked at that batch).
    Capacity = the largest predicted throughput among usable batches.
  * The operating batch starts from the batch the user entered and steps
    down the grid (128, 64, 32, 16, 8, 1) only while an SLO is missed
    (2026-11-25):
      - online  (default): requests arrive at min_throughput_rps. The
        operating batch is the largest batch <= the user's batch that meets
        the latency SLO and whose throughput >= the demand. None -> the
        instance fails (slo_impossible if no batch meets the latency SLO,
        throughput_short otherwise). Stepping down shortens T but lowers
        throughput, so it only repairs a latency miss.
      - offline: a fixed amount of work, no arrival rate. The operating batch
        is the usable batch with the largest throughput (usually the user's
        batch; a smaller one when throughput saturates or the user's batch
        misses the latency SLO). Instances compare on cost per request
        (price / throughput) and, given total_requests, time and cost to
        finish.
    No utilization headroom is applied: the instance passes when its
    predicted throughput covers the demand. demand_share = demand / capacity
    is reported so the user can see how close an instance runs to its limit
    (throughput is a prediction; the predictor reads some GPUs 10-20% fast).
  * Latency limit (resolve_latency_limit): Generate workloads take it per
    output token (TPOT-style) and the request limit is per_token x
    max_new_tokens, because the predictor's latency is the whole request
    (prefill + every generated token). Presets: 50 ms/token (real-time,
    ~20 tokens/s) and 100 ms/token (chat at reading speed, ~10 tokens/s) --
    the TPOT range LLM-serving work uses for interactive services (DistServe,
    OSDI'24). Other workloads take a per-request limit. None = no limit.
  * Each row also reports the user's own batch (input_batch_*): its P90,
    throughput and whether it meets the latency SLO, so the user sees why the
    operating batch stepped down when it did.
  * Cost is the instance's hourly price; the monthly figure is x 720
    (24 h x 30 days, same as BUDGET_HOURS_PER_MONTH in the web app) for display.
  * Budget is an upper bound. Cheapest plan = lowest hourly price (online) or
    lowest cost per request (offline) among passing instances within budget.
    Budget plan = within budget, lowest latency (online) or highest
    throughput (offline). Similar = other passing instances within
    SIMILAR_PCT of the cheapest on the same cost measure.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

BATCH_GRID = (1, 8, 16, 32, 64, 128)
LATENCY_PRESETS_MS_PER_TOKEN = {"realtime": 50.0, "standard": 100.0}
PURPOSES = ("online", "offline")
SIMILAR_PCT = 10.0
HOURS_PER_MONTH = 720.0
LATENCY_JUDGEMENT = "predicted_p90_ms"
_EPS = 1e-9


def resolve_latency_limit(slo: Mapping[str, Any] | None, max_new_tokens: int | None) -> dict[str, Any]:
    """The request-level latency limit an SLO means for this workload.

    `slo` may carry latency_per_token_ms (Generate) or max_latency_ms (per
    request); neither means no limit. Returns {"max_latency_ms": float|None,
    "basis": "per_token"|"per_request"|"none", "per_token_ms", "tokens"}.
    """
    slo = slo or {}
    per_token = _num(slo.get("latency_per_token_ms"))
    if per_token is not None:
        tokens = int(max_new_tokens) if max_new_tokens else None
        if tokens:
            return {"max_latency_ms": per_token * tokens, "basis": "per_token",
                    "per_token_ms": per_token, "tokens": tokens}
        # no generated tokens (not a Generate workload): read it per request
        return {"max_latency_ms": per_token, "basis": "per_request", "per_token_ms": None, "tokens": None}
    limit = _num(slo.get("max_latency_ms"))
    if limit is not None:
        return {"max_latency_ms": limit, "basis": "per_request", "per_token_ms": None, "tokens": None}
    return {"max_latency_ms": None, "basis": "none", "per_token_ms": None, "tokens": None}


def batch_grid(max_batch: int, grid: Iterable[int] = BATCH_GRID) -> list[int]:
    """Batch sizes to predict: the grid points not above the user's batch, plus
    the user's batch itself (VRAM was predicted for that batch only)."""
    max_batch = int(max_batch)
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    return sorted({b for b in grid if b <= max_batch} | {max_batch})


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _points(points: Iterable[Mapping[str, Any]], max_batch: int | None) -> list[dict]:
    """Predicted points as {batch, latency_ms, rps}, batch cap applied, sorted."""
    out = []
    for p in points:
        if p.get("status") != "ok":
            continue
        batch = p.get("batch")
        lat = _num(p.get(LATENCY_JUDGEMENT))
        rps = _num(p.get("predicted_throughput_rps"))
        if lat is None or rps is None or not isinstance(batch, int):
            continue
        if max_batch is not None and batch > max_batch:
            continue
        out.append({"batch": batch, "latency_ms": lat, "rps": rps})
    return sorted(out, key=lambda p: p["batch"])


def usable_points(points: Iterable[Mapping[str, Any]], max_latency_ms: float, max_batch: int | None) -> list[dict]:
    """Points with a prediction, within the latency limit and the batch cap,
    sorted by batch. Latency = predicted P90; throughput = predicted rps."""
    return [p for p in _points(points, max_batch) if p["latency_ms"] <= max_latency_ms * (1.0 + _EPS)]


def operating_point(usable: Sequence[Mapping[str, Any]], demand_rps: float) -> Mapping[str, Any] | None:
    """Online: from the user's batch down, the first usable batch whose
    throughput covers the demand (None when no usable batch does)."""
    for p in sorted(usable, key=lambda p: p["batch"], reverse=True):
        if p["rps"] >= demand_rps * (1.0 - _EPS):
            return p
    return None


def _input_batch(points: Sequence[Mapping[str, Any]], max_batch: int | None, max_latency_ms: float) -> dict[str, Any]:
    """The user's own batch (the largest predicted batch when none was given)."""
    if not points:
        return {}
    target = max_batch if max_batch is not None else max(p["batch"] for p in points)
    p = next((x for x in points if x["batch"] == target), None)
    if p is None:
        return {"input_batch": target, "input_batch_latency_ms": None, "input_batch_rps": None,
                "input_batch_latency_ok": None}
    return {"input_batch": target, "input_batch_latency_ms": p["latency_ms"], "input_batch_rps": p["rps"],
            "input_batch_latency_ok": p["latency_ms"] <= max_latency_ms * (1.0 + _EPS)}


def evaluate(
    op: Mapping[str, Any],
    capacity_rps: float,
    demand_rps: float,
    price: float | None,
    *,
    hours_per_month: float = HOURS_PER_MONTH,
) -> dict[str, Any]:
    """Online: one instance serving the whole demand at operating batch `op`."""
    return {
        "operating_batch": op["batch"],
        "latency_ms": op["latency_ms"],
        "operating_rps": op["rps"],
        "capacity_rps": capacity_rps,
        # how much of the instance's predicted capacity the demand takes
        "demand_share": demand_rps / capacity_rps,
        "cost_usd_per_hour": price,
        "cost_usd_per_month": price * hours_per_month if price is not None else None,
    }


def evaluate_offline(
    usable: Sequence[Mapping[str, Any]],
    price: float | None,
    total_requests: float | None,
    *,
    hours_per_month: float = HOURS_PER_MONTH,
) -> dict[str, Any]:
    """Offline: run at the usable batch with the largest throughput."""
    op = max(usable, key=lambda p: (p["rps"], p["batch"]))
    t, rps = op["latency_ms"], op["rps"]
    hours = total_requests / rps / 3600.0 if total_requests else None
    return {
        "operating_batch": op["batch"],
        "latency_ms": t,
        "operating_rps": rps,
        "capacity_rps": rps,
        "cost_usd_per_hour": price,
        "cost_usd_per_month": price * hours_per_month if price is not None else None,
        "cost_usd_per_million": price / (rps * 3600.0) * 1e6 if price is not None else None,
        "job_hours": hours,
        "job_cost_usd": hours * price if hours is not None and price is not None else None,
    }


def recommend(
    candidates: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    slo: Mapping[str, Any] | None,
    budget_usd_per_hour: float | None,
    *,
    max_batch: int | None = None,
    similar_pct: float = SIMILAR_PCT,
    hours_per_month: float = HOURS_PER_MONTH,
) -> dict[str, Any]:
    """One row per candidate instance plus the cheapest plan and the budget plan.

    `candidates`: instance_filter rows (instance_type, price_usd_per_hour, ...).
    `curves`: instance_type -> [{batch, status, predicted_p90_ms,
    predicted_throughput_rps, ...}] (runtime_adapter.predict_candidate_curves).
    `slo`: max_latency_ms (None = no limit), purpose ("online" default |
    "offline"), min_throughput_rps (online, required), total_requests
    (offline, optional).

    Row status: ok (meets the SLOs) / throughput_short (online: some batch
    meets the latency SLO, but none of them also covers the demand) /
    slo_impossible (no batch meets the latency SLO) / no_price /
    no_prediction.
    """
    purpose = str((slo or {}).get("purpose") or "online")
    policy = {
        "purpose": purpose,
        "latency_judgement": LATENCY_JUDGEMENT,
        "similar_pct": similar_pct,
        "hours_per_month": hours_per_month,
        "max_batch": max_batch,
        "budget_usd_per_hour": budget_usd_per_hour,
    }
    if not slo:
        return {"status": "no_slo", "policy": policy, "rows": [], "cheapest": None, "similar": [], "budget_plan": None}
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    offline = purpose == "offline"
    limit = _num(slo.get("max_latency_ms"))
    # None -> no latency limit: every predicted batch is usable.
    max_latency = limit if limit is not None else math.inf
    demand = None if offline else float(slo["min_throughput_rps"])
    total_requests = _num(slo.get("total_requests")) if offline else None
    policy.update(max_latency_ms=limit, demand_rps=demand, total_requests=total_requests)

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        name = str(cand["instance_type"])
        price = _num(cand.get("price_usd_per_hour"))
        row: dict[str, Any] = {
            "instance_type": name,
            "gpu_name": cand.get("gpu_name"),
            "price_usd_per_hour": price,
            "latency_proxy_of": None,
        }
        points = list(curves.get(name) or [])
        row["latency_proxy_of"] = next((p.get("latency_proxy_of") for p in points if p.get("latency_proxy_of")), None)
        predicted = [p for p in points if p.get("status") == "ok"]
        if not predicted:
            row.update(status="no_prediction", reason=next((p.get("error") for p in points if p.get("error")), None))
            rows.append(row)
            continue
        all_points = _points(predicted, max_batch)
        row.update(_input_batch(all_points, max_batch, max_latency))
        usable = [p for p in all_points if p["latency_ms"] <= max_latency * (1.0 + _EPS)]
        if not usable:
            fastest = min(
                (p for p in predicted if _num(p.get(LATENCY_JUDGEMENT)) is not None),
                key=lambda p: p[LATENCY_JUDGEMENT],
                default=None,
            )
            row.update(
                status="slo_impossible",
                min_latency_ms=fastest[LATENCY_JUDGEMENT] if fastest else None,
                min_latency_batch=fastest.get("batch") if fastest else None,
            )
            rows.append(row)
            continue
        best = max(usable, key=lambda p: (p["rps"], p["batch"]))
        row.update(capacity_rps=best["rps"], capacity_batch=best["batch"],
                   capacity_latency_ms=best["latency_ms"])
        if offline:
            row.update(status="ok", **evaluate_offline(usable, price, total_requests,
                                                       hours_per_month=hours_per_month))
        else:
            op = operating_point(usable, demand)
            if op is None:
                row.update(status="throughput_short", demand_share=demand / best["rps"],
                           cost_usd_per_hour=price,
                           cost_usd_per_month=price * hours_per_month if price is not None else None)
                rows.append(row)
                continue
            row.update(status="ok", **evaluate(op, best["rps"], demand, price, hours_per_month=hours_per_month))
        if price is None:
            row["status"] = "no_price"
        row["over_budget"] = (
            budget_usd_per_hour is not None and price is not None
            and price > budget_usd_per_hour * (1.0 + _EPS)
        )
        rows.append(row)

    # cost measure: hourly price (online) or cost per request (offline)
    cost_key = "cost_usd_per_million" if offline else "cost_usd_per_hour"
    ranked = [r for r in rows if r["status"] == "ok"]
    in_budget = [r for r in ranked if not r["over_budget"]]
    pool = in_budget or ranked
    cheapest = min(pool, key=lambda r: (r[cost_key], r["latency_ms"], r["instance_type"]), default=None)

    similar: list[str] = []
    if cheapest is not None:
        cap = cheapest[cost_key] * (1.0 + similar_pct / 100.0) * (1.0 + _EPS)
        similar = [r["instance_type"] for r in sorted(pool, key=lambda r: (r["latency_ms"], r[cost_key]))
                   if r is not cheapest and r[cost_key] <= cap]

    budget_plan = None
    if budget_usd_per_hour is not None and in_budget:
        if offline:
            fastest = min(in_budget, key=lambda r: (-r["operating_rps"], r[cost_key], r["instance_type"]))
        else:
            fastest = min(in_budget, key=lambda r: (round(r["latency_ms"], 9), r["cost_usd_per_hour"],
                                                     r["instance_type"]))
        keys = ["instance_type", "operating_batch", "latency_ms", "operating_rps",
                "capacity_rps", "cost_usd_per_hour", "cost_usd_per_month"]
        keys += ["cost_usd_per_million", "job_hours", "job_cost_usd"] if offline else ["demand_share"]
        budget_plan = {key: fastest[key] for key in keys}
        budget_plan["same_as_cheapest"] = cheapest is not None and fastest["instance_type"] == cheapest["instance_type"]

    def order(r: Mapping[str, Any]) -> tuple:
        rank = {"ok": 0 if not r.get("over_budget") else 1, "throughput_short": 2, "no_price": 3,
                "slo_impossible": 4, "no_prediction": 5}
        return (rank[r["status"]], r.get(cost_key) or r.get("cost_usd_per_hour") or 0.0,
                r.get("latency_ms") or r.get("min_latency_ms") or 0.0, r["instance_type"])

    rows.sort(key=order)
    if cheapest is not None:
        status = "ok" if not cheapest["over_budget"] else "over_budget"
    elif any(r["status"] == "throughput_short" for r in rows):
        status = "throughput_short"
    elif any(r["status"] == "no_price" for r in rows):
        status = "no_price"
    elif any(r["status"] == "slo_impossible" for r in rows):
        status = "slo_impossible"
    else:
        status = "no_prediction"
    return {
        "status": status,
        "policy": policy,
        "rows": rows,
        "cheapest": cheapest["instance_type"] if cheapest else None,
        "similar": similar,
        "budget_plan": budget_plan,
    }
