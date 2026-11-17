"""Instance recommendation: workload SLO -> one instance type x N units.

Turns per-instance batch curves (predicted latency/throughput at several batch
sizes) plus the user's SLO and budget into a recommendation. Pure functions,
no I/O: main.py calls `recommend()` whenever a job has both a result and an
SLO, so changing the SLO never re-runs prediction.

Decisions (throughput part, 2026-11-17/18; evidence in the `script` repo,
task 17/18 and HANDOFF §3.8/§3.9):

  * One instance type, N units. Every unit holds a full copy of the model and
    requests are split evenly across units. Mixing types is not recommended.
  * min_throughput_rps is the demand, not a per-instance pass/fail bar:
    units = ceil(demand / (capacity x target_utilization)).
  * target_utilization defaults to 0.8 (user-adjustable up to 1.0): the
    predictor reads g6e/g7/g7e 10-20% fast (measured / predicted median 1.09-1.20).
  * Latency is judged on predicted_p90_ms (the product's current judgement;
    in the agreement evaluation it overflowed least, 6.3% at target 0.8).
  * A batch is usable when its predicted P90 <= max_latency_ms and it is not
    larger than the batch the user entered (VRAM was checked at that batch).
    Capacity = the largest predicted throughput among usable batches.
  * Operating batch = the smallest usable batch whose throughput covers the
    per-unit load (demand / units). Its P90 is the shown latency T.
  * Waiting: a request can wait for the batch in flight, so the worst case is
    ~2T. Judged on T; 2T is shown ("pass" if 2T fits too, else "edge").
  * Cost is per hour (units x hourly price); the monthly figure is x 720
    (24 h x 30 days, same as BUDGET_HOURS_PER_MONTH in the web app) for display.
  * Budget is an upper bound on the whole configuration. Cheapest plan = lowest
    hourly cost within budget. Budget plan = lowest latency within budget (more
    units -> smaller operating batch -> lower latency).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

BATCH_GRID = (1, 8, 16, 32, 64, 128)
TARGET_UTILIZATION = 0.8
WAIT_FACTOR = 2.0
SIMILAR_PCT = 10.0
HOURS_PER_MONTH = 720.0
BUDGET_SEARCH_MAX_UNITS = 64
LATENCY_JUDGEMENT = "predicted_p90_ms"
_EPS = 1e-9


def batch_grid(max_batch: int, grid: Iterable[int] = BATCH_GRID) -> list[int]:
    """Batch sizes to predict: the grid points not above the user's batch, plus
    the user's batch itself (VRAM was predicted for that batch only)."""
    max_batch = int(max_batch)
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    return sorted({b for b in grid if b <= max_batch} | {max_batch})


def units_needed(demand_rps: float, capacity_rps: float, target_utilization: float) -> int:
    """ceil(demand / (capacity x target)), with a relative tolerance so an exact
    multiple does not round up to one more unit."""
    usable = capacity_rps * target_utilization
    if usable <= 0:
        raise ValueError("capacity_rps x target_utilization must be > 0")
    return max(1, math.ceil(demand_rps / usable * (1.0 - _EPS)))


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def usable_points(points: Iterable[Mapping[str, Any]], max_latency_ms: float, max_batch: int | None) -> list[dict]:
    """Points with a prediction, within the latency limit and the batch cap,
    sorted by batch. Latency = predicted P90; throughput = predicted rps."""
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
        if lat <= max_latency_ms * (1.0 + _EPS):
            out.append({"batch": batch, "latency_ms": lat, "rps": rps})
    return sorted(out, key=lambda p: p["batch"])


def operating_point(usable: Sequence[Mapping[str, Any]], load_rps: float) -> Mapping[str, Any]:
    """Smallest usable batch whose throughput covers the per-unit load."""
    for p in usable:
        if p["rps"] >= load_rps * (1.0 - _EPS):
            return p
    return max(usable, key=lambda p: p["rps"])


def evaluate(
    usable: Sequence[Mapping[str, Any]],
    units: int,
    demand_rps: float,
    price: float | None,
    max_latency_ms: float,
    *,
    wait_factor: float = WAIT_FACTOR,
    hours_per_month: float = HOURS_PER_MONTH,
) -> dict[str, Any]:
    capacity = max(p["rps"] for p in usable)
    op = operating_point(usable, demand_rps / units)
    t = op["latency_ms"]
    cost = units * price if price is not None else None
    return {
        "units": units,
        "operating_batch": op["batch"],
        "latency_ms": t,
        "wait_latency_ms": wait_factor * t,
        "verdict": "pass" if wait_factor * t <= max_latency_ms * (1.0 + _EPS) else "edge",
        "per_unit_rps": demand_rps / units,
        "capacity_rps": capacity,
        "utilization": demand_rps / (units * capacity),
        "cost_usd_per_hour": cost,
        "cost_usd_per_month": cost * hours_per_month if cost is not None else None,
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
    budget_search_max_units: int = BUDGET_SEARCH_MAX_UNITS,
) -> dict[str, Any]:
    """One row per candidate instance plus the cheapest plan and the budget plan.

    `candidates`: instance_filter rows (instance_type, price_usd_per_hour, ...).
    `curves`: instance_type -> [{batch, status, predicted_p90_ms,
    predicted_throughput_rps, ...}] (runtime_adapter.predict_candidate_curves).
    """
    policy = {
        "latency_judgement": LATENCY_JUDGEMENT,
        "target_utilization": target_utilization,
        "wait_factor": wait_factor,
        "similar_pct": similar_pct,
        "hours_per_month": hours_per_month,
        "max_batch": max_batch,
        "budget_usd_per_hour": budget_usd_per_hour,
    }
    if not slo:
        return {"status": "no_slo", "policy": policy, "rows": [], "cheapest": None, "budget_plan": None}
    if not 0 < target_utilization <= 1:
        raise ValueError(f"target_utilization must be in (0, 1], got {target_utilization}")
    max_latency = float(slo["max_latency_ms"])
    demand = float(slo["min_throughput_rps"])
    policy.update(max_latency_ms=max_latency, demand_rps=demand)

    rows: list[dict[str, Any]] = []
    feasible: dict[str, list[dict]] = {}
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
        usable = usable_points(predicted, max_latency, max_batch)
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
        capacity = max(p["rps"] for p in usable)
        units = units_needed(demand, capacity, target_utilization)
        row.update(status="ok", **evaluate(usable, units, demand, price, max_latency,
                                           wait_factor=wait_factor, hours_per_month=hours_per_month))
        row["capacity_batch"] = max(usable, key=lambda p: p["rps"])["batch"]
        row["over_budget"] = (
            budget_usd_per_hour is not None and price is not None
            and row["cost_usd_per_hour"] > budget_usd_per_hour * (1.0 + _EPS)
        )
        if price is None:
            row["status"] = "no_price"
        feasible[name] = usable
        rows.append(row)

    ranked = [r for r in rows if r["status"] == "ok"]
    in_budget = [r for r in ranked if not r["over_budget"]]
    pool = in_budget or ranked
    cheapest = min(pool, key=lambda r: (r["cost_usd_per_hour"], r["latency_ms"], r["instance_type"]), default=None)

    similar: list[str] = []
    if cheapest is not None:
        limit = cheapest["cost_usd_per_hour"] * (1.0 + similar_pct / 100.0) * (1.0 + _EPS)
        similar = [r["instance_type"] for r in pool if r is not cheapest and r["cost_usd_per_hour"] <= limit]

    budget_plan = None
    if budget_usd_per_hour is not None:
        options = []
        for r in ranked:
            price = r["price_usd_per_hour"]
            most = min(budget_search_max_units, math.floor(budget_usd_per_hour / price * (1.0 + _EPS)))
            for units in range(r["units"], most + 1):
                e = evaluate(feasible[r["instance_type"]], units, demand, price, max_latency,
                             wait_factor=wait_factor, hours_per_month=hours_per_month)
                options.append({"instance_type": r["instance_type"], **e})
        if options:
            budget_plan = min(options, key=lambda e: (round(e["latency_ms"], 9), e["utilization"],
                                                      e["cost_usd_per_hour"], e["instance_type"]))
            budget_plan["same_as_cheapest"] = (
                cheapest is not None
                and budget_plan["instance_type"] == cheapest["instance_type"]
                and budget_plan["units"] == cheapest["units"]
            )

    def order(r: Mapping[str, Any]) -> tuple:
        rank = {"ok": 0 if not r.get("over_budget") else 1, "no_price": 2, "slo_impossible": 3, "no_prediction": 4}
        return (rank[r["status"]], r.get("cost_usd_per_hour") or 0.0, r.get("latency_ms") or r.get("min_latency_ms") or 0.0,
                r["instance_type"])

    rows.sort(key=order)
    if cheapest is not None:
        status = "ok" if not cheapest["over_budget"] else "over_budget"
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
