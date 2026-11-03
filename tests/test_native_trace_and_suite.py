import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cpu_tensor_trace import selected_lifetimes, REGION


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class TraceTests(unittest.TestCase):
    def events(self):
        return [{'name':REGION,'cat':'user_annotation','ts':t,'dur':10} for t in [10,30,50]]

    def mem(self,t,size,address=1):
        return {'name':'[memory]','cat':'cpu_instant_event','ts':t,
                'args':{'Device Type':0,'Addr':address,'Bytes':size}}

    def test_data_loader_excluded_free_outside_region_kept(self):
        events=self.events()+[self.mem(1,100,2),self.mem(2,-100,2),self.mem(11,16),self.mem(25,-16)]
        blocks,_=selected_lifetimes(events)
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0][1]['ts'],25)

    def test_live_cache_not_silently_freed_at_region_end(self):
        blocks,_=selected_lifetimes(self.events()+[self.mem(11,64)])
        self.assertIsNone(blocks[0][1])

    def test_bad_memory_pair_is_an_error(self):
        with self.assertRaisesRegex(ValueError,'size mismatch'):
            selected_lifetimes(self.events()+[self.mem(11,64),self.mem(12,-32)])

    def test_missing_measurement_is_an_error(self):
        with self.assertRaises(ValueError):selected_lifetimes(self.events()[:2])


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner=load('suite','tools/run_prediction_suite.py')
        cls.plan=json.loads((ROOT/'benchmarks/aws342/selected-plan.json').read_text())

    def test_342_requests_keep_shape_dtype_and_generation_mode(self):
        cases=self.runner.select_cases(self.plan,'all')
        self.assertEqual(len(cases),342)
        for case in cases:
            args=self.runner.request(case,{'model_type':'test'},4)
            self.assertEqual(args['dtype'],case['dtype'])
            for k,v in case['params'].items():self.assertEqual(args[k],v)
            self.assertEqual(args['mode'],'prefill' if case['workload_type']=='vision_cls' else 'generate')

    def test_baseline9_subset(self):
        cases=self.runner.select_cases(self.plan,'baseline9')
        self.assertEqual(len(cases),9)
        self.assertTrue(all(c['dtype']=='fp16' and c['case_id'].endswith('__01') for c in cases))

    def test_heldout_cannot_change_fitted_constant(self):
        calibration=load('calibration','tools/calibrate_gpu_overheads.py')
        def row(number,value,dtype='fp16'):
            return {'case':{'case_id':f'model{number}__{dtype}__01','dtype':dtype},
                'environment':{'gpu':'NVIDIA L4',**{k:'same' for k in calibration.SCOPE_KEYS}},
                'memory_samples':[{'nvml_process_bytes':value+100,'reserved_bytes':100}]}
        fit=Path('fit.tar.gz');valid=Path('validation.tar.gz')
        def source(path):
            return iter([(row(i,200),'sha') for i in range(9)] if path==fit else [(row(1,900,'fp32'),'sha2')])
        with patch.object(calibration,'memory_cases',side_effect=source):
            profiles,report=calibration.build([fit],[valid])
        self.assertEqual(profiles['gpus']['NVIDIA L4']['outside_median_bytes'],200)
        self.assertEqual(report['heldout'][0]['median_error_bytes'],-700)


if __name__=='__main__':unittest.main()
