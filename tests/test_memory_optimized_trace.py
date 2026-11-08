import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference_trace_runtime as runtime
from cpu_tensor_trace import REGION, selected_lifetimes

spec = importlib.util.spec_from_file_location('recovery', ROOT/'tools/run_prediction_recovery.py')
sys.path.insert(0, str(ROOT/'tools'))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class TraceReaderTests(unittest.TestCase):
    def read(self, value, chunk=7, raw=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.json'
            path.write_text(value if raw else json.dumps(value, ensure_ascii=False))
            return list(runtime.iter_trace_events(path, chunk))

    def test_chunk_boundaries_and_metadata(self):
        events = [{'name': '한글 " \\ [memory]', 'ts': 12345.6789, 'args': {'s': 'x'*300}}]
        value = {'metadata': {'nested': ['traceEvents', 123456789]}, 'traceEvents': events, 'tail': [False, None]}
        for chunk in (1, 2, 7, 64, 4096):
            self.assertEqual(self.read(value, chunk), events)

    def test_empty_trace(self):
        self.assertEqual(self.read({'traceEvents': []}), [])

    def test_invalid_trace_rejected(self):
        for value in ('{}', '{"traceEvents":[', '{"traceEvents":[{}',
                      '{"traceEvents":[{},]}', '{"traceEvents":[3]}',
                      '{"traceEvents":[],"traceEvents":[]}', '{"traceEvents":[]}junk'):
            with self.assertRaises(ValueError, msg=value):
                self.read(value, raw=True)

    def test_filter_keeps_exact_lifetime_order(self):
        def mem(ts, size, address):
            return {'name': '[memory]', 'cat': 'cpu_instant_event', 'ts': ts,
                    'args': {'Device Type': 0, 'Bytes': size, 'Addr': address}}
        events = [{'name': REGION, 'cat': 'user_annotation', 'ts': t, 'dur': 10} for t in (10, 30, 50)]
        events += [mem(11, 4096, 1), mem(22, -4096, 1), mem(31, 2048, 1),
                   mem(32, -2048, 1), mem(51, 1024, 2),
                   {'name': 'aten::mm', 'cat': 'cpu_op', 'ts': 31}]
        random.Random(4).shuffle(events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.json'
            path.write_text(json.dumps({'traceEvents': events}))
            filtered, total = runtime.read_memory_events(path, REGION)
        self.assertEqual(total, len(events))
        self.assertEqual(selected_lifetimes(filtered, 3), selected_lifetimes(events, 3))


class RecoveryTests(unittest.TestCase):
    def classify(self, text, code, state=None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'case.log';log.write_text(text)
            return recovery.classify(code, state or {'ExitCode': code, 'OOMKilled': False}, log, **kwargs)

    def test_exit_without_json_does_not_discard_returncode(self):
        result = self.classify('', 137)
        self.assertEqual(result['exit_code'], 137)
        self.assertFalse(result['oom_killed'])
        self.assertEqual(result['error_kind'], 'process_exit_without_json')

    def test_only_docker_oom_evidence_is_classified_as_oom(self):
        result = self.classify('', 137, {'ExitCode': 137, 'OOMKilled': True})
        self.assertEqual(result['error_kind'], 'cgroup_oom')

    def test_cancel_and_timeout_are_distinct(self):
        self.assertEqual(self.classify('', 137, cancelled=True)['error_kind'], 'cancelled')
        self.assertEqual(self.classify('', 137, timed_out=True)['error_kind'], 'timeout')

    def test_nonzero_exit_cannot_be_success(self):
        result = self.classify(json.dumps({'status': 'ok'}), 1)
        self.assertEqual(result['status'], 'error')

    def test_success_preserves_both_byte_results(self):
        result = self.classify(json.dumps({'status': 'ok', 'predicted_allocated_bytes': 512,
                                         'predicted_reserved_bytes': 2097152}), 0)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual((result['predicted_allocated_bytes'], result['predicted_reserved_bytes']), (512, 2097152))

    def test_last_phase_survives_missing_json(self):
        result = self.classify('CGF_PROGRESS {"event":"phase_start","phase":"allocator_simulation"}\n', 137)
        self.assertEqual(result['stage'], 'allocator_simulation')

    def test_container_has_hard_limit_and_readonly_sources(self):
        command = recovery.job_command('image', '/host/work', '/host/cache', 32*recovery.GIB, 4)
        self.assertEqual(command[command.index('--memory')+1], str(32*recovery.GIB))
        self.assertEqual(command[command.index('--memory-swap')+1], str(32*recovery.GIB))
        self.assertEqual(command[command.index('--network')+1], 'none')
        self.assertTrue(all(s.endswith(',readonly') for s in command if s.startswith('type=bind,')))

    def test_remaining_selection_does_not_rerun_successes(self):
        plan = json.loads((ROOT/'benchmarks/aws342/selected-plan.json').read_text())
        case = plan['cases'][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/(case['case_id']+'.prediction.json')).write_text('{"status":"ok"}')
            cases = recovery.choose_cases(plan, None, path)
        self.assertEqual(len(cases), 341)
        self.assertNotIn(case['case_id'], {c['case_id'] for c in cases})

    def test_unknown_case_rejected(self):
        plan = json.loads((ROOT/'benchmarks/aws342/selected-plan.json').read_text())
        with self.assertRaises(ValueError):
            recovery.choose_cases(plan, ['unknown'], None)

    def test_reference_comparison_uses_exact_bytes_and_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'case.input.json').write_text(json.dumps({'request': {'threads': 4}}))
            (root/'case.prediction.json').write_text(json.dumps({'status':'ok',
                'predicted_allocated_bytes':512, 'predicted_reserved_bytes':2097152}))
            result = {'status':'ok','predicted_allocated_bytes':513,'predicted_reserved_bytes':2097152}
            comparison = recovery.compare_reference('case', {'threads':4}, result, root)
            self.assertEqual(comparison['status'], 'mismatch')
            self.assertEqual(comparison['delta_bytes']['predicted_allocated_bytes'], 1)
            self.assertEqual(recovery.compare_reference('case', {'threads':1}, result, root)['status'], 'incompatible_input')

    def test_reference_revision_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'case.input.json').write_text(json.dumps({'revision':'old','request':{}}))
            result = {'status':'ok','revision':'old','predicted_allocated_bytes':512,'predicted_reserved_bytes':1024}
            (root/'case.prediction.json').write_text(json.dumps(result))
            self.assertEqual(recovery.compare_reference('case', {}, result, root, 'new')['status'], 'incompatible_input')

    def test_timeout_stops_and_removes_job_container(self):
        proc = Mock()
        proc.communicate.side_effect = [subprocess.TimeoutExpired('docker', 1), ('', '')]
        proc.poll.side_effect = [None, 137]
        proc.returncode = 137
        def fake_docker(*args, **kwargs):
            return types.SimpleNamespace(stdout=(json.dumps([{'Id':'test-id','HostConfig':{'Memory':1024,'MemorySwap':1024},'Mounts':[]}]) if args == ('inspect', 'unique-job') else json.dumps({'ExitCode': 137, 'OOMKilled': False})) if args[0]=='inspect' else '')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(recovery, 'docker', side_effect=fake_docker) as calls, \
             patch.object(recovery, 'stop_job') as stop, \
             patch.object(recovery.subprocess, 'Popen', return_value=proc):
            result = recovery.run_job('unique-job', ['image'], {}, Path(directory)/'log', 1)
        self.assertEqual(result['error_kind'], 'timeout')
        stop.assert_called_once_with('unique-job')
        self.assertEqual(calls.call_args.args, ('rm', '-f', 'unique-job'))


@unittest.skipUnless(os.environ.get('XMEM_REFERENCE_ROOT'), 'Set XMEM_REFERENCE_ROOT to the pinned upstream xMem checkout')
class AllocatorEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the actual pinned allocator/profiler modules without importing the
        # unrelated training estimator from perf_estimator/__init__.py.
        cls.previous = {k: v for k, v in sys.modules.items() if k == 'perf_estimator' or k.startswith('perf_estimator.')}
        for k in cls.previous:
            del sys.modules[k]
        package = types.ModuleType('perf_estimator')
        package.__path__ = [str(Path(os.environ['XMEM_REFERENCE_ROOT'])/'perf_estimator')]
        sys.modules['perf_estimator'] = package

    @classmethod
    def tearDownClass(cls):
        for k in list(sys.modules):
            if k == 'perf_estimator' or k.startswith('perf_estimator.'):
                del sys.modules[k]
        sys.modules.update(cls.previous)

    def test_pinned_allocator_full_and_summary_match_every_counter(self):
        from perf_estimator.allocator import AllocatorSim
        from perf_estimator.config import Config
        blocks = [types.SimpleNamespace(bytes=513, alloc_time=0, free_time=None)]
        rng = random.Random(9)
        for i in range(400):
            blocks.append(types.SimpleNamespace(bytes=rng.choice([1, 511, 512, 513, 1000000, 2097153, 12000000]),
                                                alloc_time=i*2+1, free_time=i*2+rng.randint(1, 20)))
        with tempfile.TemporaryDirectory() as directory:
            conf = Config(save2tmp=True, run_id='equivalence-'+Path(directory).name)
            expected = AllocatorSim(max_allocated_memory_gb=2, config=conf).simulate(blocks)
            self.assertFalse(expected.oom)
            actual = runtime.simulate(blocks, 2, conf)
        self.assertEqual(actual._trace.max_usage_changes, expected._trace.max_usage_changes)
        self.assertEqual(actual._trace.max_segment_changes, expected._trace.max_segment_changes)
        self.assertGreater(len(expected._trace._trace_history), 0)
        self.assertEqual(actual._trace._trace_history, [])

    def test_capacity_failure_is_identified(self):
        from perf_estimator.config import Config
        block = types.SimpleNamespace(bytes=32*1024**2, alloc_time=0, free_time=None)
        with self.assertRaisesRegex(RuntimeError, 'simulated_gpu_capacity_exceeded'):
            runtime.simulate([block], 0.001, Config(save2tmp=True, run_id='capacity-test'))

    def test_allocator_bug_is_not_mislabeled_as_capacity(self):
        from perf_estimator.config import Config
        from perf_estimator.allocator.allocator import CachingAllocator
        block = types.SimpleNamespace(bytes=512, alloc_time=0, free_time=None)
        with patch.object(CachingAllocator, 'malloc', side_effect=ValueError('invalid state')):
            with self.assertRaisesRegex(RuntimeError, 'allocator_simulation_failed: ValueError'):
                runtime.simulate([block], 2, Config(save2tmp=True, run_id='failure-test'))

    def test_explicit_legacy_invocations_remain_available(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from predict_vram import storage_sizes
        from cpu_tensor_trace import trace_and_estimate
        torch.set_num_threads(1)
        config=GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=64,n_positions=64,
                          eos_token_id=63,bos_token_id=63,pad_token_id=63)
        config._attn_implementation='eager'
        model=GPT2LMHeadModel(config).eval()
        inputs={'input_ids':torch.ones((1,8),dtype=torch.long),'attention_mask':torch.ones((1,8),dtype=torch.long)}
        with patch.object(model,'generate',wraps=model.generate) as generate:
            result=trace_and_estimate(model,inputs,'generate',4,storage_sizes,2,
                                      warmup_invocations=1,measured_invocations=3)
        self.assertEqual(generate.call_count,4)
        self.assertEqual(result['trace_summary']['measured_invocations'],3)
        self.assertEqual(result['trace_summary']['warmup_invocations'],1)

    def test_real_tiny_gpt2_trace_replay_matches_upstream(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from perf_estimator.allocator import AllocatorSim
        from perf_estimator.config import Config
        from perf_estimator.profiler import MemoryBlock, CpuInstantNode
        from predict_vram import storage_sizes
        from cpu_tensor_trace import trace_and_estimate
        torch.manual_seed(20261006)
        torch.set_num_threads(1)
        config = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=64, n_positions=64,
                            eos_token_id=63, bos_token_id=63, pad_token_id=63)
        config._attn_implementation = 'eager'
        model = GPT2LMHeadModel(config).eval()
        inputs = {'input_ids': torch.ones((1, 8), dtype=torch.long), 'attention_mask': torch.ones((1, 8), dtype=torch.long)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.json'
            with patch.object(model, 'generate', wraps=model.generate) as generate:
                optimized = trace_and_estimate(model, inputs, 'generate', 4, storage_sizes, 2, trace_output=path)
                self.assertEqual(generate.call_count, 1)
            original_events = json.loads(path.read_text())['traceEvents']
            lifetimes, regions = selected_lifetimes(original_events)
            blocks = []
            for alloc, free in lifetimes:
                block = MemoryBlock(CpuInstantNode(alloc))
                if free is not None:
                    block.set_free_node(CpuInstantNode(free))
                blocks.append(block)
            resident = storage_sizes(list(model.parameters())+list(model.buffers())) + storage_sizes(list(inputs.values()))
            for i, size in enumerate(resident):
                if size:
                    event = {'ph':'i','cat':'cpu_instant_event','s':'t','name':'[memory]',
                             'pid':0,'tid':0,'ts':regions[0][0]-1,
                             'args':{'Bytes':size,'Addr':-i-1,'Device Id':-1,'Device Type':0,'Ev Idx':0}}
                    blocks.append(MemoryBlock(CpuInstantNode(event)))
            blocks.sort(key=lambda b:b.alloc_time)
            expected = AllocatorSim(max_allocated_memory_gb=2, config=Config(save2tmp=True,run_id='tiny-original')).simulate(blocks)
        self.assertFalse(expected.oom)
        self.assertEqual(optimized['memory']['tensor'], max(expected._trace.max_usage_changes))
        self.assertEqual(optimized['memory']['segment'], max(expected._trace.max_segment_changes))
        self.assertEqual(optimized['trace_summary']['measured_invocations'], 1)
        self.assertEqual(optimized['trace_summary']['warmup_invocations'], 0)


if __name__ == '__main__':
    unittest.main()
