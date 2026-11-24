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
  * A batch is usable when its predicted P90 <= the latency SLO and it is not
    larger than the batch the user entered (VRAM was checked at that batch).
    Capacity = the largest predicted throughput among usable batches.
  * Latency T = the predicted P90 of ONE batch run (batch processing time).
    It does not include time spent waiting in a queue or for a batch to fill;
    the latency definition is the latency part's (predictor) and is kept as is.
    2T (a request that just missed the batch in flight waits one more run,
    Nexus SOSP'19 §4.1) is shown for reference, not judged.
  * Two purposes (2026-11-24):
      - online  (default): requests arrive at min_throughput_rps. The
        instance passes when demand <= capacity x target_utilization, and
        the operating batch is the smallest usable batch whose throughput x
        target_utilization covers the demand -- so the headroom holds at the
        batch the instance actually runs, and a lower target moves it to a
        larger batch. Smaller batch = shorter T.
      - offline: a fixed amount of work, no arrival rate. The operating batch
        is the usable batch with the largest throughput (start from the
        user's batch, step down while it misses the latency SLO), and
        instances compare on cost per request (price / throughput) and,
        given total_requests, the time and cost to finish.
  * target_utilization ("목표 부하율" on screen = demand / throughput, not GPU
    utilization) defaults to 0.8, user-adjustable 0.5-0.9. It is headroom
    for prediction error and traffic swings: the predictor reads
    g6e/g7/g7e 10-20% fast (measured / predicted median 1.09-1.20), and
    a demand figure is an average that peaks above. It does not change T.
  * Latency limit (resolve_latency_limit): Generate workloads take it per
    output token (TPOT-style) and the request limit is per_token x
    max_new_tokens, because the predictor's latency is the whole request
    (prefill + every generated token). Presets: 50 ms/token (real-time,
    ~20 tokens/s) and 100 ms/token (chat at reading speed, ~10 tokens/s) --
    the TPOT range LLM-serving work uses for interactive services (DistServe,
    OSDI'24). Other workloads take a per-request limit. None = no limit.
  * Each row also reports the user's own batch (input_batch_*): its P90,
    throughput and whether it meets the latency SLO, so the user sees how the
    batch they entered behaves even when another batch is chosen.
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
TARGET_UTILIZATION = 0.8
TARGET_UTILIZATION_MIN = 0.5
TARGET_UTILIZATION_MAX = 0.9
LATENCY_PRESETS_MS_PER_TOKEN = {"realtime": 50.0, "standard": 100.0}
PURPOSES = ("online", "offline")
WAIT_FACTOR = 2.0
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


def operating_point(usable: Sequence[Mapping[str, Any]], load_rps: float,
                    target_utilization: float = 1.0) -> Mapping[str, Any]:
    """Smallest usable batch whose throughput x target covers the load."""
    for p in usable:
        if p["rps"] * target_utilization >= load_rps * (1.0 - _EPS):
            return p
    return max(usable, key=lambda p: p["rps"])


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
    usable: Sequence[Mapping[str, Any]],
    demand_rps: float,
    price: float | None,
    max_latency_ms: float,
    *,
    target_utilization: float = TARGET_UTILIZATION,
    wait_factor: float = WAIT_FACTOR,
    hours_per_month: float = HOURS_PER_MONTH,
) -> dict[str, Any]:
    """Online: one instance serving the whole demand (caller checked capacity)."""
    capacity = max(p["rps"] for p in usable)
    op = operating_point(usable, demand_rps, target_utilization)
    t = op["latency_ms"]
    return {
        "operating_batch": op["batch"],
        "latency_ms": t,
        "wait_latency_ms": wait_factor * t,
        "verdict": "pass" if wait_factor * t <= max_latency_ms * (1.0 + _EPS) else "edge",
        "operating_rps": op["rps"],
        "capacity_rps": capacity,
        # share of the operating batch's throughput the demand takes (<= target)
        "load_ratio": demand_rps / op["rps"],
        "cost_usd_per_hour": price,
        "cost_usd_per_month": price * hours_per_month if price is not None else None,
    }


def evaluate_offline(
    usable: Sequence[Mapping[str, Any]],
    price: float | None,
    max_latency_ms: float,
    total_requests: float | None,
    *,
    wait_factor: float = WAIT_FACTOR,
    hours_per_month: float = HOURS_PER_MONTH,
) -> dict[str, Any]:
    """Offline: run at the usable batch with the largest throughput."""
    op = max(usable, key=lambda p: (p["rps"], -p["batch"]))
    t, rps = op["latency_ms"], op["rps"]
    hours = total_requests / rps / 3600.0 if total_requests else None
    return {
        "operating_batch": op["batch"],
        "latency_ms": t,
        "wait_latency_ms": wait_factor * t,
        "verdict": "pass" if wait_factor * t <= max_latency_ms * (1.0 + _EPS) else "edge",
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
    target_utilization: float = TARGET_UTILIZATION,
    wait_factor: float = WAIT_FACTOR,
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

    Row status: ok (meets the SLOs) / throughput_short (online: latency OK,
    but even its best usable batch x target < demand) / slo_impossible (no
    batch meets the latency SLO) / no_price / no_prediction.
    """
    purpose = str((slo or {}).get("purpose") or "online")
    policy = {
        "purpose": purpose,
        "latency_judgement": LATENCY_JUDGEMENT,
        "target_utilization": target_utilization,
        "wait_factor": wait_factor,
        "similar_pct": similar_pct,
        "hours_per_month": hours_per_month,
        "max_batch": max_batch,
        "budget_usd_per_hour": budget_usd_per_hour,
    }
    if not slo:
        return {"status": "no_slo", "policy": policy, "rows": [], "cheapest": None, "similar": [], "budget_plan": None}
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    if not 0 < target_utilization <= 1:
        raise ValueError(f"target_utilization must be in (0, 1], got {target_utilization}")
    offline = purpose == "offline"
    limit = _num(slo.get("max_latency_ms"))
    # None -> no latency limit: every predicted batch is usable, verdict "pass".
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
        best = max(usable, key=lambda p: (p["rps"], -p["batch"]))
        row.update(capacity_rps=best["rps"], capacity_batch=best["batch"],
                   capacity_latency_ms=best["latency_ms"])
        if offline:
            row.update(status="ok", **evaluate_offline(usable, price, max_latency, total_requests,
                                                       wait_factor=wait_factor, hours_per_month=hours_per_month))
        else:
            # the demand this instance can take at the target load
            row["max_demand_rps"] = best["rps"] * target_utilization
            if demand > best["rps"] * target_utilization * (1.0 + _EPS):
                row.update(status="throughput_short", cost_usd_per_hour=price,
                           cost_usd_per_month=price * hours_per_month if price is not None else None)
                rows.append(row)
                continue
            row.update(status="ok", **evaluate(usable, demand, price, max_latency,
                                               target_utilization=target_utilization,
                                               wait_factor=wait_factor, hours_per_month=hours_per_month))
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
            fastest = min(in_budget, key=lambda r: (round(r["latency_ms"], 9), r["load_ratio"],
                                                     r["cost_usd_per_hour"], r["instance_type"]))
        keys = ["instance_type", "operating_batch", "latency_ms", "wait_latency_ms", "verdict", "operating_rps",
                "capacity_rps", "cost_usd_per_hour", "cost_usd_per_month"]
        keys += ["cost_usd_per_million", "job_hours", "job_cost_usd"] if offline else ["load_ratio"]
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
