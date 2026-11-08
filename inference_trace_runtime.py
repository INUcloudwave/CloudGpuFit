"""Bounded JSON reading, phase diagnostics and optional xMem history suppression.

Allocator transitions and byte counters remain the pinned upstream implementation.
No global monkey patches or new runtime dependencies are used.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import sys
import time
import os


LAST_PHASE = None


def resource_snapshot():
    result = {"pid": os.getpid(), "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_bytes'] = int(line.split()[1]) * 1024
    except OSError:
        pass
    for filename in ('memory.current', 'memory.peak'):
        try:
            result['cgroup_' + filename.replace('.', '_') + '_bytes'] = int((Path('/sys/fs/cgroup') / filename).read_text())
        except (OSError, ValueError):
            pass
    return result


def progress(event, **fields):
    print('CGF_PROGRESS ' + json.dumps({"event": event, "unix_time": time.time(), **resource_snapshot(), **fields}),
          file=sys.stderr, flush=True)


@contextmanager
def phase(name):
    global LAST_PHASE
    LAST_PHASE = name
    start = time.monotonic()
    progress('phase_start', phase=name)
    try:
        yield
    except BaseException as exc:
        progress('phase_error', phase=name, elapsed_seconds=time.monotonic()-start, exception_type=type(exc).__name__)
        raise
    else:
        progress('phase_end', phase=name, elapsed_seconds=time.monotonic()-start)


class _JSONStream:
    def __init__(self, source, chunk_size):
        self.source, self.chunk_size = source, chunk_size
        self.buffer, self.pos, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def more(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        chunk = self.source.read(self.chunk_size)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ''
            self.more()

    def expect(self, char):
        if self.peek() != char:
            raise ValueError(f'Invalid Chrome trace JSON: expected {char!r}')
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # Do not accept a partial scalar at a chunk boundary.
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError('Truncated or malformed Chrome trace JSON') from None
                self.more()


def iter_trace_events(path, chunk_size=65536):
    """Yield traceEvents in file order; metadata is not mistaken for events."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    with Path(path).open(encoding='utf-8') as source:
        reader = _JSONStream(source, chunk_size)
        reader.expect('{')
        found = False
        if reader.peek() != '}':
            while True:
                key = reader.value()
                if not isinstance(key, str):
                    raise ValueError('Chrome trace object key must be a string')
                reader.expect(':')
                if key == 'traceEvents':
                    if found:
                        raise ValueError('Duplicate traceEvents array')
                    found = True
                    reader.expect('[')
                    if reader.peek() != ']':
                        while True:
                            event = reader.value()
                            if not isinstance(event, dict):
                                raise ValueError('Chrome trace event must be an object')
                            yield event
                            if reader.peek() != ',':
                                break
                            reader.expect(',')
                    reader.expect(']')
                else:
                    reader.value()
                if reader.peek() != ',':
                    break
                reader.expect(',')
        reader.expect('}')
        if reader.peek():
            raise ValueError('Trailing data in Chrome trace JSON')
        if not found:
            raise ValueError('Chrome trace has no traceEvents array')


def read_memory_events(path, region):
    selected, total = [], 0
    for event in iter_trace_events(path):
        total += 1
        if event.get('name') == region and event.get('cat') == 'user_annotation':
            selected.append(event)
        elif event.get('name') == '[memory]' and event.get('cat') == 'cpu_instant_event':
            args = event.get('args', {})
            if args.get('Device Type') == 0 and args.get('Bytes', 0) != 0:
                selected.append(event)
    return selected, total


def simulate(blocks, gpu_memory_gb, config, *, record_history=False):
    from perf_estimator.allocator import AllocatorSim
    from perf_estimator.allocator.allocator import CachingAllocator, Trace

    class SummaryTrace(Trace):
        def _record(self, block, action, use_size=True):
            # Keep inherited segment/usage counters and all allocator transitions.
            # Only visualization/debug snapshots and formatted messages are omitted.
            pass

    class ReportingAllocator(CachingAllocator):
        failure = None

        def malloc(self, *args, **kwargs):
            try:
                return super().malloc(*args, **kwargs)
            except Exception as exc:
                self.failure = exc
                raise

        def free(self, *args, **kwargs):
            try:
                return super().free(*args, **kwargs)
            except Exception as exc:
                self.failure = exc
                raise

    simulator = AllocatorSim(max_allocated_memory_gb=gpu_memory_gb, config=config)
    allocator = ReportingAllocator(allowed_memory_maximum=simulator._allocator.allowed_memory_maximum)
    if not record_history:
        allocator._trace = SummaryTrace(allocator._segments)
    simulator._allocator = allocator
    result = simulator.simulate(blocks)
    if result.oom:
        failure = result.failure
        if isinstance(failure, MemoryError) and str(failure).startswith('OOM when allocating '):
            raise RuntimeError('simulated_gpu_capacity_exceeded: ' + str(failure)) from failure
        if failure is not None:
            raise RuntimeError(f'allocator_simulation_failed: {type(failure).__name__}: {failure}') from failure
        raise RuntimeError('allocator_simulation_failed: upstream simulator rejected an event; not a confirmed capacity OOM')
    return result
