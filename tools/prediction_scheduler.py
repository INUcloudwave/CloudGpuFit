"""Admit concurrent predictions using observed host RAM, not case reservations."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import threading
import time

GIB = 1024 ** 3


def host_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(value.split()[0]) * 1024
    total, available = values['MemTotal'], values['MemAvailable']
    return {'total_bytes': total, 'available_bytes': available,
            'used_bytes': total - available}


def admission_allowed(memory, threshold_bytes, reserve_bytes):
    return memory['used_bytes'] <= threshold_bytes and memory['available_bytes'] >= reserve_bytes


def run_cases(cases, execute, record, *, jobs, threshold_bytes, reserve_bytes,
              poll_seconds=1, launch_interval_seconds=5, memory_reader=host_memory,
              notify=lambda event, **fields: None):
    """Only this thread admits/writes results; workers own their job containers.

    record() returns False to stop admission (e.g. an exact-reference mismatch).
    Cancellation reaches every worker, then all completed/cancelled results drain.
    The threshold is an admission rule, not a cap on future aggregate RAM growth.
    """
    pending = list(cases)
    active = {}
    cancel = threading.Event()
    interrupted = validation_failed = False
    last_launch = float('-inf')
    waiting = None
    pool = ThreadPoolExecutor(max_workers=jobs)
    try:
        while pending or active:
            try:
                for future in [f for f in active if f.done()]:
                    case = active.pop(future)
                    result = future.result()
                    if record(case, result) is False:
                        validation_failed = True
                        cancel.set()
                    notify('finished', case_id=case['case_id'],
                           running=[c['case_id'] for c in active.values()], pending=len(pending))
                if cancel.is_set():
                    pending.clear()
                if pending and len(active) < jobs:
                    memory = memory_reader()
                    allowed = admission_allowed(memory, threshold_bytes, reserve_bytes)
                    staggered = active and time.monotonic()-last_launch < launch_interval_seconds
                    reason = 'memory' if not allowed else 'launch_interval' if staggered else None
                    if reason is None:
                        case = pending.pop(0)
                        future = pool.submit(execute, case, cancel)
                        active[future] = case
                        last_launch = time.monotonic()
                        waiting = None
                        notify('start', case_id=case['case_id'], memory=memory,
                               running=[c['case_id'] for c in active.values()], pending=len(pending))
                    elif waiting != reason:
                        waiting = reason
                        notify('wait', reason=reason, memory=memory,
                               running=[c['case_id'] for c in active.values()], pending=len(pending))
                if active:
                    wait(active, timeout=poll_seconds, return_when=FIRST_COMPLETED)
                elif pending:
                    time.sleep(poll_seconds)
            except KeyboardInterrupt:
                interrupted = True
                cancel.set()
                pending.clear()
                notify('cancellation_requested', running=[c['case_id'] for c in active.values()], pending=0)
    finally:
        cancel.set()
        pool.shutdown(wait=True)
    return {'interrupted': interrupted, 'validation_failed': validation_failed}
