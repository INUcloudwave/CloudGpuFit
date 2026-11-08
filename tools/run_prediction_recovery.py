#!/usr/bin/env python3
"""Run selected CPU predictions sequentially in disposable, memory-limited containers.

The prepared xmem-cpu environment is snapshotted locally. Its mounted source and
all old results remain untouched. Each invocation requires a fresh output path.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from run_prediction_suite import ROOT, CONFIG, prediction_json, request, save, select_cases

SOURCES = ('predict_vram.py', 'cpu_tensor_trace.py', 'inference_trace_runtime.py',
           'vram_policy.py', 'gpu_overheads.json')
GIB = 1024 ** 3


def docker(*args, **kwargs):
    return subprocess.run(['docker', *args], check=True, text=True, encoding='utf-8',
                          errors='replace', capture_output=True, **kwargs)


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot determine host MemAvailable')


def last_phase(log):
    result = None
    with Path(log).open(errors='replace') as source:
        for line in source:
            if line.startswith('CGF_PROGRESS '):
                try:
                    result = json.loads(line[len('CGF_PROGRESS '):]).get('phase', result)
                except ValueError:
                    pass
    return result


def classify(exit_code, state, log, *, timed_out=False, cancelled=False):
    """Return process evidence even when no predictor JSON was emitted."""
    result = {'status': 'error', 'exit_code': exit_code,
              'container_exit_code': state.get('ExitCode'),
              'oom_killed': bool(state.get('OOMKilled')), 'stage': last_phase(log)}
    if cancelled:
        result.update(error_kind='cancelled', error='Prediction cancelled; job container stopped')
    elif timed_out:
        result.update(error_kind='timeout', error='Prediction exceeded configured timeout')
    elif result['oom_killed']:
        result.update(error_kind='cgroup_oom', error='Job container exceeded its memory limit')
    else:
        try:
            parsed = prediction_json(Path(log).read_text(errors='replace'))
        except ValueError:
            result.update(error_kind='missing_prediction_json' if exit_code == 0 else 'process_exit_without_json',
                          error=f'Predictor JSON missing; docker exit={exit_code}; OOMKilled={result["oom_killed"]}')
        else:
            result.update(parsed)
            # Runtime evidence is authoritative even if a stale/misleading JSON says ok.
            result.update(exit_code=exit_code, container_exit_code=state.get('ExitCode'),
                          oom_killed=bool(state.get('OOMKilled')))
            if exit_code or state.get('ExitCode', 0):
                result.update(status='error', error_kind='process_exit',
                              error=parsed.get('error', f'Predictor process exited unsuccessfully: {exit_code}'))
            elif parsed['status'] != 'ok':
                result['error_kind'] = 'predictor_error'
    return result


def stop_job(name):
    try:
        docker('stop', '--time', '3', name, timeout=10)
    except (subprocess.SubprocessError, OSError):
        # Only the uniquely named container created for this attempt is removed.
        docker('kill', name, timeout=15)


def run_job(name, command, payload, log, timeout):
    """The container PID 1 is the predictor, so Docker can attribute OOM kills."""
    proc = None
    timed_out = cancelled = False
    state = {}
    created = False
    try:
        docker('create', '--name', name, *command, timeout=30)
        created = True
        info = json.loads(docker('inspect', name, timeout=15).stdout)[0]
        metadata = {'container_id': info['Id'], 'container_name': name,
                    'memory_limit_bytes': info['HostConfig']['Memory'],
                    'memory_swap_bytes': info['HostConfig']['MemorySwap'], 'mounts': info['Mounts'],
                    'nano_cpus': info['HostConfig'].get('NanoCpus'),
                    'network_mode': info['HostConfig'].get('NetworkMode')}
        with Path(log).open('x') as output:
            proc = subprocess.Popen(['docker', 'start', '-a', '-i', name], stdin=subprocess.PIPE,
                                    stdout=output, stderr=subprocess.STDOUT, text=True)
            try:
                for _ in range(20):
                    active = json.loads(docker('inspect', '--format', '{{json .State}}', name, timeout=15).stdout)
                    host_pid = active.get('Pid', 0)
                    if host_pid:
                        metadata['host_pid'] = host_pid
                        try:
                            metadata['host_cgroup'] = Path(f'/proc/{host_pid}/cgroup').read_text()
                        except OSError:
                            pass
                        break
                    if 'Running' not in active or proc.poll() is not None:
                        break
                    time.sleep(0.05)
                proc.communicate(json.dumps(payload), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_job(name)
            except KeyboardInterrupt:
                cancelled = True
                stop_job(name)
            finally:
                if proc.poll() is None:
                    try:
                        proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
        state = json.loads(docker('inspect', '--format', '{{json .State}}', name, timeout=15).stdout)
        result = classify(proc.returncode, state, log, timed_out=timed_out, cancelled=cancelled)
        result.update(container=metadata, container_state=state)
        return result
    finally:
        # Also cover errors while inspecting or starting the Docker CLI.
        try:
            if created:
                docker('rm', '-f', name, timeout=15)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def job_command(image, work, cache, memory_bytes, threads):
    args = ['-i', '--entrypoint', 'python', '--network', 'none', '--memory', str(memory_bytes),
            '--memory-swap', str(memory_bytes), '--cpus', str(threads),
            '--mount', f'type=bind,src={work},dst=/work,readonly',
            '--mount', f'type=bind,src={cache},dst=/root/.cache/huggingface,readonly']
    # A separate directory avoids creating missing mountpoints under read-only /work.
    args += ['--mount', f'type=bind,src={ROOT},dst=/recovery,readonly']
    for value in (f'OMP_NUM_THREADS={threads}', f'MKL_NUM_THREADS={threads}',
                  'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
                  'PYTHONPATH=/recovery:/work/xMem'):
        args += ['-e', value]
    return args + ['-w', '/work/xMem', image, '-u', '/recovery/predict_vram.py', '--json', '-']


def choose_cases(plan, ids, remaining_from):
    cases = select_cases(plan, 'all')
    if ids:
        selected = set(ids)
        unknown = selected - {c['case_id'] for c in cases}
        if unknown:
            raise ValueError('Unknown case IDs: ' + ', '.join(sorted(unknown)))
        cases = [c for c in cases if c['case_id'] in selected]
    elif remaining_from:
        def succeeded(case):
            path = remaining_from / (case['case_id'] + '.prediction.json')
            return path.exists() and json.loads(path.read_text()).get('status') == 'ok'
        cases = [c for c in cases if not succeeded(c)]
    else:
        raise ValueError('Specify --case-id for validation or --remaining-from after validation')
    return cases


def compare_reference(case_id, payload, result, root, revision=None):
    if root is None or result.get('status') != 'ok':
        return {'status': 'not_requested' if root is None else 'prediction_failed'}
    prediction = root / (case_id + '.prediction.json')
    inputs = root / (case_id + '.input.json')
    if not prediction.exists() or not inputs.exists():
        return {'status': 'no_successful_reference'}
    old = json.loads(prediction.read_text())
    if old.get('status') != 'ok':
        return {'status': 'no_successful_reference'}
    old_input = json.loads(inputs.read_text())
    if old_input.get('request') != payload or (revision is not None and
            (old_input.get('revision') != revision or old.get('revision') != revision)):
        return {'status': 'incompatible_input'}
    names = ('predicted_allocated_bytes', 'predicted_reserved_bytes')
    deltas = {name: result[name] - old[name] for name in names}
    return {'status': 'equal' if all(v == 0 for v in deltas.values()) else 'mismatch',
            'delta_bytes': deltas, 'reference': str(prediction.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--case-id', action='append', help='Exact case ID; repeat to select multiple cases')
    selection.add_argument('--remaining-from', type=Path, help='Read-only old results; select every non-successful case')
    parser.add_argument('--container', default='xmem-cpu')
    parser.add_argument('--snapshot-image', help='Reuse an explicit local snapshot after verifying its Python environment')
    parser.add_argument('--compare-with', type=Path, help='Read-only old results for exact allocated/reserved comparison')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--memory-gib', type=float, default=32)
    parser.add_argument('--host-reserve-gib', type=float, default=6)
    parser.add_argument('--timeout-seconds', type=float, default=3600)
    args = parser.parse_args()
    if args.threads < 1 or args.memory_gib <= 0 or args.host_reserve_gib < 0 or args.timeout_seconds <= 0:
        parser.error('Invalid resource limits')
    out = args.output_root.resolve()
    if out.exists():
        parser.error('Use a new output directory; existing results are never overwritten')
    plan = json.loads((ROOT/'benchmarks/aws342/selected-plan.json').read_text())
    pins = json.loads((ROOT/'benchmarks/aws342/revisions.json').read_text())
    cases = choose_cases(plan, args.case_id, args.remaining_from)
    if not cases:
        print('No incomplete cases selected')
        return 0
    lock = Path(f'/tmp/cloudgpufit-cpu-recovery-{os.getuid()}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Do not snapshot an actively computing or stopped environment.
    info = json.loads(docker('inspect', args.container, timeout=15).stdout)[0]
    if not info['State']['Running']:
        raise RuntimeError('The prepared CPU container must be running')
    top = docker('top', args.container, '-eo', 'pid,args', timeout=15).stdout
    if 'predict_vram.py' in top or 'run_prediction' in top:
        raise RuntimeError('A prediction is still running; existing work was left untouched')
    mounts = {m['Destination']: m['Source'] for m in info['Mounts'] if m['Type'] == 'bind'}
    work, cache = mounts['/work'], mounts['/root/.cache/huggingface']
    allocator_lock = json.loads((ROOT/'benchmarks/aws342/xmem-allocator-lock.json').read_text())
    check = 'import json,hashlib,pathlib,sys; names=json.load(sys.stdin); print(json.dumps({n:hashlib.sha256((pathlib.Path("/work/xMem")/n).read_bytes()).hexdigest() for n in names}))'
    hashes = json.loads(docker('exec', '-i', args.container, 'python', '-c', check,
                              input=json.dumps(list(allocator_lock['sha256'])), timeout=30).stdout)
    if hashes != allocator_lock['sha256']:
        raise RuntimeError('Prepared xMem allocator differs from pinned source')
    envcode = 'import json,torch,transformers,sys; print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,python=sys.version.split()[0],cuda=torch.version.cuda,gpu_available=torch.cuda.is_available())))'
    environment = json.loads(docker('exec', args.container, 'python', '-c', envcode, timeout=60).stdout)
    if (environment['torch'], environment['transformers'], environment['python'], environment['cuda'], environment['gpu_available']) != ('2.6.0+cu124', '4.52.4', '3.11.11', '12.4', False):
        raise RuntimeError('Prepared CPU environment differs from pinned versions: ' + str(environment))
    if args.snapshot_image:
        snapshot_environment = json.loads(docker('run', '--rm', '--network', 'none', '--memory', str(2*GIB),
            '--memory-swap', str(2*GIB), '--entrypoint', 'python', args.snapshot_image, '-c', envcode, timeout=60).stdout)
        if snapshot_environment != environment:
            raise RuntimeError('Snapshot environment differs from the prepared container')
    configs = {model: json.loads(docker('exec', args.container, 'python', '-c', CONFIG,
                                      model, pins[model]['revision'], timeout=120).stdout)
               for model in dict.fromkeys(c['model'] for c in cases)}
    memory_bytes = int(args.memory_gib * GIB)
    required = memory_bytes + int(args.host_reserve_gib * GIB)
    if available_bytes() < required:
        raise RuntimeError('Host MemAvailable is below memory limit + safety reserve; no job started')
    token = uuid.uuid4().hex
    image = 'cloudgpufit-cpu-recovery:' + token
    out.mkdir(parents=True)
    git_commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'status': 'preparing', 'git_commit': git_commit, 'total_cases': len(cases), 'completed_cases': 0, 'success_count': 0,
                'environment': environment, 'base_image_id': info['Image'], 'allocator_lock': allocator_lock,
                'identity': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                'plan_sha256': hashlib.sha256((ROOT/'benchmarks/aws342/selected-plan.json').read_bytes()).hexdigest(),
                'revisions_sha256': hashlib.sha256((ROOT/'benchmarks/aws342/revisions.json').read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'threads': args.threads, 'memory_limit_bytes': memory_bytes, 'host_reserve_bytes': required-memory_bytes,
                'timeout_seconds': args.timeout_seconds, 'jobs': 1,
                'reused_snapshot': args.snapshot_image,
                'selection': [c['case_id'] for c in cases], 'started_unix': time.time()}
    save(out/'manifest.json', manifest)
    start = time.monotonic()
    snapshot = False
    interrupted = False
    validation_failed = False
    try:
        if args.snapshot_image:
            image_id = docker('image', 'inspect', '--format', '{{.Id}}', args.snapshot_image, timeout=15).stdout.strip()
        else:
            docker('commit', '--pause=false', args.container, image, timeout=900)
            snapshot = True
            # Some Docker versions emit deprecation warnings on stdout.
            image_id = docker('image', 'inspect', '--format', '{{.Id}}', image, timeout=15).stdout.strip()
        manifest.update(status='running', snapshot_image_id=image_id)
        save(out/'manifest.json', manifest)
        for number, case in enumerate(cases, 1):
            cid = case['case_id']
            payload = request(case, configs[case['model']], args.threads)
            save(out/(cid+'.input.json'), {'case_id': cid, 'revision': pins[case['model']]['revision'], 'request': payload})
            print(f'START {cid}', flush=True)
            case_start = time.monotonic()
            try:
                if available_bytes() < required:
                    raise RuntimeError('Host MemAvailable fell below admission threshold')
                result = run_job('cgf-cpu-' + token[:12] + '-' + str(number),
                                 job_command(image_id, work, cache, memory_bytes, args.threads),
                                 payload, out/(cid+'.log'), args.timeout_seconds)
            except KeyboardInterrupt:
                result = {'status': 'error', 'error_kind': 'cancelled', 'error': 'Recovery interrupted'}
            except Exception as exc:
                result = {'status': 'error', 'error_kind': 'runner_error', 'error': f'{type(exc).__name__}: {exc}'}
                if isinstance(exc, subprocess.CalledProcessError):
                    result['docker_stderr'] = (exc.stderr or '')[-4000:]
            result.update(case_id=cid, revision=pins[case['model']]['revision'],
                          runner_elapsed_seconds=time.monotonic()-case_start)
            comparison = compare_reference(cid, payload, result, args.compare_with, pins[case['model']]['revision'])
            result['reference_comparison'] = comparison
            save(out/(cid+'.prediction.json'), result)
            manifest['completed_cases'] += 1
            manifest['success_count'] += result['status'] == 'ok'
            manifest['elapsed_seconds'] = time.monotonic()-start
            save(out/'manifest.json', manifest)
            print(f'[{number}/{len(cases)}] {result["status"]} {cid} {result.get("error_kind", "")}', flush=True)
            if result.get('error_kind') == 'cancelled':
                interrupted = True
                break
            if comparison['status'] in ('mismatch', 'incompatible_input'):
                print('STOP reference comparison: ' + json.dumps(comparison), flush=True)
                validation_failed = True
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if snapshot:
            subprocess.run(['docker', 'image', 'rm', image], capture_output=True, timeout=60)
        manifest.update(status='interrupted' if interrupted else 'validation_mismatch' if validation_failed else
                        'complete' if manifest['completed_cases'] == len(cases) and manifest['success_count'] == len(cases)
                        else 'complete_with_errors' if manifest['completed_cases'] == len(cases) else 'runner_failed',
                        elapsed_seconds=time.monotonic()-start)
        save(out/'manifest.json', manifest)
    return 130 if interrupted else 2 if validation_failed else 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    raise SystemExit(main())
